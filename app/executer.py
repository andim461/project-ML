import PyInstaller.__main__

if __name__ == '__main__':
    source = './main.py'
    flags = ['--onefile', '--noconsole']

    PyInstaller.__main__.run([source, *flags])