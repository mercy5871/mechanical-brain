from random import random
import numpy as np
import math
import gym
from gym import spaces
# a = spaces.Discrete(2)
# scope = 60
# b = spaces.Box(np.ones(scope * (3 + 1)) * -1, np.ones(scope * (3 + 1)))
# print(np.array(b))
# np.ones(scope * (3 + 1)) * -1
#
# from keras.models import Model as m
# m.predict()

class MarketEnv(gym.Env):

	PENALTY = 0.997

	def __init__(self, dir_path, target_codes, input_codes, start_date, end_date, scope = 60, sudden_death = -1, cumulative_reward = False):
		self.startDate = start_date
		self.endDate = end_date
		self.scope = scope
		self.last_action = 0
		self.sudden_death = sudden_death
		self.cumulative_reward = cumulative_reward

		self.inputCodes = []
		self.targetCodes = []
		self.dataMap = {}

		for code in (target_codes + input_codes):
			fn = dir_path + code + ".csv"

			data = {}
			lastClose = 0
			lastVolume = 0
			try:
				f = open(fn, "r")
				for line in f:
					if line.strip() != "":
						dt, openPrice, high, low, close, volume = line.strip().split(",")
						# print(dt, openPrice, high, low, close, volume)
						try:
							if dt >= start_date:
								high = float(high) if high != "" else float(close)
								low = float(low) if low != "" else float(close)
								close = float(close)
								volume = float(volume)

								if lastClose > 0 and close > 0 and lastVolume > 0:
									close_ = (close - lastClose) / lastClose
									high_ = (high - close) / close
									low_ = (low - close) / close
									volume_ = (volume - lastVolume) / lastVolume

									data[dt] = (high_, low_, close_, volume_)

								lastClose = close
								lastVolume = volume
						except Exception as e:
							print(e, line.strip().split(","))
				f.close()
			except Exception as e:
				print(e)

			if len(data.keys()) > scope:
				self.dataMap[code] = data
				if code in target_codes:
					self.targetCodes.append(code)
				if code in input_codes:
					self.inputCodes.append(code)

		self.actions = ["SHORT", "LONG"]

		self.action_space = spaces.Discrete(len(self.actions))
		self.observation_space = spaces.Box(np.ones(scope * (len(input_codes) + 1)) * -1, np.ones(scope * (len(input_codes) + 1)))

		self.reset()
		self._seed()

	def _step(self, action):
		if self.done:
			return self.state, self.reward, self.done, {}

		self.reward = 0
		if action == 2:
			self.boughts.append(0.0)
		elif self.actions[action] == "LONG":
			if sum(self.boughts) < 0:
				for b in self.boughts:
					self.reward += -(b + 1)
				if self.cumulative_reward:
					self.reward = self.reward / max(1, len(self.boughts))

				if self.sudden_death * len(self.boughts) > self.reward:
					self.done = True

				self.boughts = []

			self.boughts.append(1.0)
		elif self.actions[action] == "SHORT":
			if sum(self.boughts) > 0:
				for b in self.boughts:
					self.reward += b - 1
				if self.cumulative_reward:
					self.reward = self.reward / max(1, len(self.boughts))

				if self.sudden_death * len(self.boughts) > self.reward:
					self.done = True

				self.boughts = []

			self.boughts.append(-1.0)
		else:
			pass

		vari = self.target[self.targetDates[self.currentTargetIndex]][2]
		self.cum = self.cum * (1 + vari)
		# print('vari:',vari)
		# print('boughts:','\n',self.boughts)

		if action == 2 and self.actions[self.last_action] == "SHORT" and vari <= 0:
			self.rat = self.rat * (1 - vari)
		elif action == 2 and self.actions[self.last_action] == "SHORT" and vari >= 0:
			self.rat = self.rat * (1 - vari)
		elif action == 2 and self.actions[self.last_action] == "LONG" and vari <= 0:
			self.rat = self.rat * (1 + vari)
		elif action == 2 and self.actions[self.last_action] == "LONG" and vari >= 0:
			self.rat = self.rat * (1 + vari)
		elif self.actions[action] == "SHORT" and vari <= 0:
			self.rat = self.rat * (1 - vari)
		elif self.actions[action] == "SHORT" and vari > 0:
			self.rat = self.rat * (1 - vari)
		elif self.actions[action] == "LONG" and vari > 0:
			self.rat = self.rat * (1 + vari)
		elif self.actions[action] == "LONG" and vari <= 0:
			self.rat = self.rat * (1 + vari)
		else:
			pass

		for i in np.arange(len(self.boughts)):
			self.boughts[i] = self.boughts[i] * MarketEnv.PENALTY * (1 + vari * (-1 if sum(self.boughts) < 0 else 1))

		self.defineState()
		self.currentTargetIndex += 1
		if self.currentTargetIndex >= len(self.targetDates) or self.endDate <= self.targetDates[self.currentTargetIndex]:
			self.done = True

		if self.done:
			for b in self.boughts:
				self.reward += (b * (1 if sum(self.boughts) > 0 else -1)) - 1
			if self.cumulative_reward:
				self.reward = self.reward / max(1, len(self.boughts))

			self.boughts = []

		return self.state, self.reward, self.done, {"dt": self.targetDates[self.currentTargetIndex], "cum": self.cum, "rat": self.rat, "code": self.targetCode}

	def _reset(self):
		self.targetCode = self.targetCodes[int(random() * len(self.targetCodes))]
		self.target = self.dataMap[self.targetCode]
		self.targetDates = sorted(self.target.keys())
		self.currentTargetIndex = self.scope
		self.boughts = []
		self.cum = 1.
		self.rat = 1.

		self.done = False
		self.reward = 0

		self.defineState()

		return self.state

	def _render(self, mode='human', close=False):
		if close:
			return
		return self.state

	'''
	def _close(self):
		pass

	def _configure(self):
		pass
	'''

	def _seed(self):
		return int(random() * 100)

	def defineState(self):
		tmpState = []

		budget = (sum(self.boughts) / len(self.boughts)) if len(self.boughts) > 0 else 1.
		size = math.log(max(1., len(self.boughts)), 100)
		position = 1. if sum(self.boughts) > 0 else 0.
		tmpState.append([[budget, size, position]])

		subject = []
		subjectVolume = []
		for i in np.arange(self.scope):
			try:
				subject.append([self.target[self.targetDates[self.currentTargetIndex - 1 - i]][2]])
				subjectVolume.append([self.target[self.targetDates[self.currentTargetIndex - 1 - i]][3]])
			except Exception as e:
				print(self.targetCode, self.currentTargetIndex, i, len(self.targetDates))
				self.done = True
		tmpState.append([[subject, subjectVolume]])

		tmpState = [np.array(i) for i in tmpState]
		self.state = tmpState
