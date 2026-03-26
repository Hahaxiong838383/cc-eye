#!/bin/bash
# start-daemon.sh — 通过 Terminal.app 启动 cc-eye daemon（绕过 macOS TCC 摄像头限制）
# launchd 进程没有摄像头权限，但 Terminal.app 有。
# 用 osascript 让 Terminal 执行脚本，摄像头权限随 Terminal 走。

DAEMON_CMD="cd ~/mycc/2-Projects/cc-eye && source .venv/bin/activate && python camera_daemon.py"

# 检查 daemon 是否已在运行
if pgrep -f "python camera_daemon.py" > /dev/null 2>&1; then
    echo "[cc-eye] daemon 已在运行"
    exit 0
fi

# 通过 Terminal.app 启动，获得摄像头权限
osascript <<EOF
tell application "Terminal"
    -- 打开新 tab（如果有窗口）或新窗口
    if (count of windows) > 0 then
        tell application "System Events" to keystroke "t" using command down
        delay 0.5
        do script "$DAEMON_CMD" in front window
    else
        do script "$DAEMON_CMD"
    end if
    -- 最小化窗口（不打扰）
    delay 1
    set miniaturized of front window to true
end tell
EOF

echo "[cc-eye] daemon 已通过 Terminal 启动（窗口已最小化）"
