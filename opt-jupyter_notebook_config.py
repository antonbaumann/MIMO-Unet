import os
c.NotebookApp.allow_root = True
c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.port = int(os.getenv('JUPYTER_PORT','8888'))
c.NotebookApp.custom_display_url = 'http://hostname:%d' % (c.NotebookApp.port)
c.NotebookApp.open_browser = False
c.NotebookApp.terminado_settings = { 'shell_command': [ '/bin/bash', '-i'] }
