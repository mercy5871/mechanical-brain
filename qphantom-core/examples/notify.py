from QPhantom.core.notify import MailNotifier

noti = MailNotifier.default(to='duanhong@ppmoney.com', project='Test Project')

# noti.logger.info("test info")

with noti.guardian('Test yag'):

    noti.set_project('r')
    noti.set_receiver('duanhong@ppmoney.com')
    noti.logger.info("test yag")
    raise Exception('9')

# with noti.guardian('Test with label'):
#     noti.logger.info("test_info3")
#     1 + 1
