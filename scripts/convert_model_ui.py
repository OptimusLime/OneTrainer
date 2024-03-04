from onetrainer.modules.ui.ConvertModelUI import ConvertModelUI
import os
import sys

sys.path.append(os.getcwd())


def main():
    ui = ConvertModelUI(None)
    ui.mainloop()


if __name__ == '__main__':
    main()
