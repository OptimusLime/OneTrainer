from onetrainer.modules.ui.TrainUI import TrainUI
import os
import sys

sys.path.append(os.getcwd())


def main():
    ui = TrainUI()
    ui.mainloop()
    ui.close()


if __name__ == '__main__':
    main()
