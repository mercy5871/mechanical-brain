import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import h5py
import numpy as np
import tensorflow as tf
import keras
import seaborn as sn
from matplotlib import pyplot as plt
from market_env import MarketEnv
from market_model_builder import MarketPolicyGradientModelBuilder
from model_builder import AbstractModelBuilder
from Metrics import Metrics
M = Metrics()

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class PolicyGradient:

	def __init__(self, env, discount = 0.99, model_filename = None, history_filename = None):
		self.env = env
		self.discount = discount
		self.model_filename = model_filename
		self.history_filename = history_filename

		from keras.optimizers import SGD
		self.model = MarketPolicyGradientModelBuilder(modelFilename).getModel()
		# self.model = MarketPolicyGradientModelBuilder().buildModel()
		sgd = SGD(lr = 0.1, decay = 1e-6, momentum = 0.9, nesterov = True)
		self.model.compile(loss='mse', optimizer='rmsprop')

	def discount_rewards(self, r):
		discounted_r = np.zeros_like(r)
		running_add = 0
		r = r.flatten()

		for t in reversed(np.arange(0, r.size)):
			if r[t] != 0:
				running_add = 0

			running_add = running_add * self.discount + r[t]
			discounted_r[t] = running_add

		return discounted_r

	def train(self, max_episode = 1e1, max_path_length = 200, threshold = 0.5, verbose = 0):
		env = self.env
		model = self.model
		avg_reward_sum = 0.

		for e in np.arange(max_episode):
			env.reset()
			observation = env.reset()
			# print('observation[0].shape:', '\n', observation[0].shape)
			# print('observation[1].shape:', '\n', observation[1].shape)
			# print('observation[1]:', '\n', observation[1])

			game_over = False
			reward_sum = 0
			last_y = np.array([0, 1])

			inputs = []
			outputs = []
			predicteds = []
			rewards = []
			count = 0
			date_list = []
			value_list = []
			benchmark_list = []
			predict_summary = []
			while not game_over:
				# count += 1
				# print('count:',count)
				aprob = model.predict(observation)[0]
				# print('aprob:', '\n', aprob)
				# print('aprob_shape:', '\n', aprob.shape)
				# print('aprob[0]:', '\n', aprob[0])
				# print('aprob[1]:', '\n', aprob[1])
				inputs.append(observation)
				predicteds.append(aprob)

				if aprob.shape[0] > 1:
					if max(aprob) > threshold:
						action = np.argsort(aprob)[-1]
						# action = np.random.choice(self.env.action_space.n, 1, p = aprob / np.sum(aprob))[0]
						self.env.last_action = action
						y = np.zeros([self.env.action_space.n])
						y[action] = 1.
						# print('action:', action)
						# print('y:', y)
						last_y = y.copy()
						outputs.append(y)
					else:
						action = 2
						outputs.append(last_y)
				else:
					action = 0 if np.random.uniform() < aprob else 1

					y = [float(action)]
					outputs.append(y)

				predict_summary.append(max(aprob))
				observation, reward, game_over, info = self.env.step(action)
				# print('boservation[0]:',observation[0])
				reward_sum += float(reward)
				#print('reward_sum:','\n',reward_sum)
				rewards.append(float(reward))
				#print('rewards:','\n',rewards)
				date_list.append(info["dt"])
				value_list.append(info["rat"])
				benchmark_list.append(info["cum"])

				if verbose > 0 :
					if action == 2:
						color = bcolors.OKBLUE if aprob[0] == max(aprob) else bcolors.FAIL
						print("%s:\t%s\t%.2f\t%.2f\t%.2f\t" % (info["dt"], color + "HOLD!!!" + bcolors.ENDC, reward_sum, info["cum"], info["rat"]) + ("\t".join(["%s:%.2f" % (l, i) for l, i in zip(env.actions, aprob.tolist())])))
					elif env.actions[action] == "LONG" or env.actions[action] == "SHORT":
						color = bcolors.FAIL if env.actions[action] == "LONG" else bcolors.OKBLUE
						print("%s:\t%s\t%.2f\t%.2f\t%.2f\t" % (info["dt"], color + env.actions[action] + bcolors.ENDC, reward_sum, info["cum"], info["rat"]) + ("\t".join(["%s:%.2f" % (l, i) for l, i in zip(env.actions, aprob.tolist())])))

			avg_reward_sum = avg_reward_sum * 0.99 + reward_sum * 0.01
			fc = bcolors.FAIL if info["cum"] >= 1 else bcolors.OKBLUE
			fr = bcolors.FAIL if info["rat"] >= 1 else bcolors.OKBLUE
			bw = bcolors.ENDC
			toPrint = "%d\t\t%s\t%.2f\t%s\t%s\t%.2f" % (e, info["code"], reward_sum, fc+("%.2f" % info["cum"])+bw, fr+("%.2f" % info["rat"])+bw, avg_reward_sum)
			# toPrint = "%d\t\t%s\t%s\t%.2f\t%.2f\t%.2f" % (e, info["code"], (bcolors.FAIL if reward_sum >= 0 else bcolors.OKBLUE) + ("%.2f" % reward_sum) + bcolors.ENDC, info["cum"], info["rat"], avg_reward_sum)
			if self.history_filename != None:
				os.system("echo %s >> %s" % (toPrint, self.history_filename))
			print(toPrint)
			# print('avg_reward_sum', '\n', avg_reward_sum)
			# print('env.actions:',env.actions)
			M.plot_trade_summary(indices=date_list, value=value_list, benchmark=benchmark_list)
			plt.hist(predict_summary,bins=200)
			plt.show()

			dim = len(inputs[0])
			inputs_ = [[] for i in np.arange(dim)]
			for obs in inputs:
				for i, block in enumerate(obs):
					inputs_[i].append(block[0])
			inputs_ = [np.array(inputs_[i]) for i in np.arange(dim)]
			#print('inputs:', '\n', inputs[0][1])
			outputs_ = np.vstack(outputs)
			# print('outputs_:', '\n', outputs_)
			predicteds_ = np.vstack(predicteds)
			rewards_ = np.vstack(rewards)

			discounted_rewards_ = self.discount_rewards(rewards_)
			#discounted_rewards_ -= np.mean(discounted_rewards_)
			discounted_rewards_ /= np.std(discounted_rewards_)

			#outputs_ *= discounted_rewards_
			for i, r in enumerate(zip(rewards, discounted_rewards_)):
				reward, discounted_reward = r

				if verbose > 1:
					print (outputs_[i],)

				#outputs_[i] = 0.5 + (2 * outputs_[i] - 1) * discounted_reward
				if discounted_reward < 0:
					outputs_[i] = 1 - outputs_[i]
					outputs_[i] = outputs_[i] / sum(outputs_[i])
				outputs_[i] = np.minimum(1, np.maximum(0, predicteds_[i] + (outputs_[i] - predicteds_[i]) * abs(discounted_reward)))

				if verbose > 1:
					print(predicteds_[i], outputs_[i], reward, discounted_reward)


			# print('inputs_:', '\n', inputs_[0].shape)
			# print('inputs_:', '\n', inputs_[1].shape)
			# print('inputs_:', '\n', inputs_[1])
			# print('outputs_:', '\n', outputs_)
			# print('layers:', '\n', model.layers)
			model.fit(inputs_, outputs_, epochs = 1, verbose = 0, shuffle = True)
			model.save_weights('model_1.h5')

if __name__ == "__main__":
	# import sys
	# import codecs
    #keras.layers.Convolution2D()

    historyFilename =   None
    modelFilename   =   None # 'model_1.h5' # None

    name = 'SP500_'
    codeList = []
    for i in range(1, 10):
        codeList.append(name + str(i))

    # codeList =  ['SP500'] # ['DJI','SP500','NASDAQ','005380','005930','005935','012330','015760','028260','032830','035420'] # '090430' '000660' ['DJI','SP500']

    env = MarketEnv(dir_path="/home/mercy/notebook/sample_data/", target_codes=codeList, input_codes=[], start_date="2009-01-03", end_date="2016-01-03", sudden_death=-1)
    pg = PolicyGradient(env, discount=0.9, model_filename=modelFilename, history_filename=historyFilename) # start_date="2009-01-03", end_date="2016-01-03"
    pg.train(max_episode = 50, threshold = 0.5, verbose = 0)


#%%
import numpy as np
import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt

# DJI.csv     SP500.csv     NASDAQ.csv
index_name = 'SP500'
random_edg = 0.015

data = pd.read_csv('~/notebook/sample_data/%s.csv'%index_name,names=['open','high','low','close','vol'])
data.index.name = 'date'
data_norm = data.copy()
for c in data_norm.columns:
    data_norm[c] = data[c]/data[c][0]

for i in range(1, 10):
    data_number = str(i)
    lt = []
    for i in range(data_norm.shape[0] * data_norm.shape[1]):
        lt.append(np.random.uniform(-random_edg, random_edg))
    random_lt = np.array(lt).reshape(data_norm.shape[0], data_norm.shape[1])
    random_lt[0] = [0,0,0,0,0]
    data_create = data_norm + random_lt
    data_norm
    data_create

    plt.figure(1)
    plt.subplot(211)
    plt.plot(data_norm['close'].values[2000:2100])
    plt.subplot(212)
    plt.plot(data_create['close'].values[2000:2100])
    plt.title(data_number)
    plt.show()

    data_create.reset_index('trade_date',drop = False,inplace = True)
    data_create.to_csv('/home/mercy/notebook/sample_data/%s_'%index_name + data_number + '.csv',index=False,header=False)

#%%
import numpy as np
import pandas as pd

# '.DJI.N', '.INX.A', '.IXIC.O'   DJI SP500 NASDAQ
PATH_USHIS = '/home/data/ushis/'
ushis_all = pd.read_pickle(PATH_USHIS + 'ushis_all.pkl')

# ushis_all['wd_open'] = ushis_all['wd_open']*ushis_all['wd_div_x']
# ushis_all['wd_close'] = ushis_all['wd_close']*ushis_all['wd_div_x']
# ushis_all['wd_high'] = ushis_all['wd_high']*ushis_all['wd_div_x']
# ushis_all['wd_low'] = ushis_all['wd_low']*ushis_all['wd_div_x']

groupby_data_code = ushis_all.groupby(level='unique_code')
data = groupby_data_code.get_group('.IXIC.O')[['wd_open','wd_high','wd_low','wd_close','wd_vol']]
data.reset_index('unique_code',drop = True,inplace = True)
data.reset_index('trade_date',drop = False,inplace = True)

data['trade_date'] = [d[:4]+'-'+d[4:6]+'-'+d[6:] for d in data['trade_date']]
data = data.fillna(method='ffill')
data.to_csv('~/notebook/sample_data/NASDAQ.csv',index=False,header=False)
data.shape
