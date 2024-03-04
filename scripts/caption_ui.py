from onetrainer.modules.ui.CaptionUI import CaptionUI
from onetrainer.modules.util.args.CaptionUIArgs import CaptionUIArgs
import os
import sys

sys.path.append(os.getcwd())


def main():
    args = CaptionUIArgs.parse_args()

    ui = CaptionUI(None, args.dir, args.include_subdirectories)
    ui.mainloop()


if __name__ == '__main__':
    main()
