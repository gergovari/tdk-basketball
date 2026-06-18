import sys, select, time
import termios, tty

old = termios.tcgetattr(sys.stdin)
tty.setcbreak(sys.stdin.fileno())
try:
    for i in range(10):
        print(f"Loop {i}")
        if select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            print(f"Key pressed: {ch}")
        time.sleep(0.1)
finally:
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
