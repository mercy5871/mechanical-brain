import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

s_from = "admin@q-phantom.com"
s_to = "admin@q-phantom.com"
smtp = smtplib.SMTP("smtp.q-phantom.com", port=25)
smtp.login(s_from, "ftloap")

msg = MIMEMultipart()
msg['Subject'] = "Hello world!"
msg['From'] = s_from
msg['To'] = s_to
msg.attach(MIMEText("showmethemoney"))

smtp.sendmail(s_from, s_to, msg.as_string())
smtp.quit()
