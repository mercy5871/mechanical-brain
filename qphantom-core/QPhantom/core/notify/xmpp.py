# %%

import threading
import sleekxmpp
from QPhantom.core.notify.notifier import Notifier

class XMPPBot(sleekxmpp.ClientXMPP):
    def __init__(self, jid, password):
        super(XMPPBot, self).__init__(jid, password)
        self.ready = threading.Event()
        self.register_plugin('xep_0030')
        self.register_plugin('xep_0066') # OOB
        self.register_plugin('xep_0231') # BOB
        self.add_event_handler('session_start', self.session_start)

    def session_start(self, event):
        self.get_roster()
        self.send_presence()
        self.ready.set()

    def send_image_html(self, jid, img_url):
        m = self.Message()
        m['to'] = jid
        m['type'] = 'chat'
        m['body'] = 'Tried sending an image using HTML-IM'
        m['html']['body'] = '<img src="%s" />' % img_url
        m.send()

    def send_image_bob(self, jid, img, name=None):
        m = self.Message()
        m['to'] = jid
        m['type'] = 'chat'
        if img:
            cid = self['xep_0231'].set_bob(img, 'image/png')
            m['body'] = name if name is not None else "YOUR CLIENT MAY NOT SUPPORT IMAGE"
            m['html']['body'] = '<img src="cid:%s" />' % cid
            m.send()

    def send_image_oob(self, jid, img_url):
        m = self.Message()
        m['to'] = jid
        m['type'] = 'chat'
        m['body'] = 'Tried sending an image using OOB'
        m['oob']['url'] = img_url
        m.send()


class XMPPNotifier(Notifier):
    '''
    目前只能用于文本内容的发送
    password 是发送邮箱的授权码，用来替代登录密码
    smtpserver 是发送邮箱的发送服务器
    eg：
        sender = 'q_phantom@163.com'
        receiver = 'duanhong@ppmoney.com'
        smtpserver = 'smtp.163.com'
        password = 'qphantom1'
    '''
    def __init__(self, bot, to=None, project=None):
        super(XMPPNotifier, self).__init__(project)
        self.bot = bot
        self.to = to

    @staticmethod
    def default(to=None, project=None):
        bot = XMPPBot('qphantom.notify@notify.q-phantom.com', 'qphantom1')
        bot.connect()
        bot.process(block=False)
        bot.ready.wait()
        return XMPPNotifier(
            bot=bot,
            to=to,
            project=project
        )

    @staticmethod
    def ashares(to=None, project=None):
        bot = XMPPBot('ashares_log@notify.q-phantom.com', 'ashares1')
        bot.connect()
        bot.process(block=False)
        bot.ready.wait()
        return XMPPNotifier(
            bot=bot,
            to=to,
            project=project
        )

    def set_receiver(self, to):
        self.to = to

    def image(self, img, name=None):
        self.bot.send_image_bob(self.to, img, name=name)

    def send(self, subject=None, content=None):
        assert(self.to is not None)
        log_buffer = self.log_buffer
        self.log_buffer = list()
        if subject is not None:
            self.bot.send_message(self.to, subject)
        if content is not None:
            self.bot.send_message(self.to, content)
        for i, c in enumerate(log_buffer):
            if isinstance(c, dict):
                if c["type"] == "img":
                    self.bot.send_image_bob(self.to, c["body"], name=c["name"])
                else:
                    self.bot.send_message(self.to, f"UNKNOWN MSG with type: {type(c['body'])}")
            else:
                self.bot.send_message(self.to, c)

    def emit(self, record):
        msg = self.format(record)
        self.bot.send_message(self.to, msg)

if __name__ == "__main__":
    noti = XMPPNotifier.default(to='earthson@notify.q-phantom.com', project="test")
    noti.info("TEST_LABEL", "INFO TEST")
    noti.send("heh", "AAA")
    noti.logger.info("XXXXXX")
    noti.logger.info("Test")
    with open("/Users/earthson/Pictures/Avator/error.png", "rb") as f:
        noti.image(f.read())
    noti.send()
