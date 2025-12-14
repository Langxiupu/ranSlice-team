#!/bin/bash

# 启动 tmux 会话并运行调试程序
tmux new-session -d -s dreamer_session

# 激活环境并启动 Python 程序
tmux send-keys -t dreamer_session "conda activate dreamer" C-m
tmux send-keys -t dreamer_session "python /home/ps/proj/zbm/ranslice/dreamer.py --configs ranSlice --task ranSlice_task --logdir /home/ps/proj/zbm/ranslice/logdir/test" C-m
# 查看 tmux 输出
tmux attach -t dreamer_session
