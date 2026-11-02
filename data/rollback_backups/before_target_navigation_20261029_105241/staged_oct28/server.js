const WebSocket = require('ws');

const host = process.env.WS_HOST || '127.0.0.1';
const port = Number.parseInt(process.env.WS_PORT || '8766', 10);
const wss = new WebSocket.Server({ host, port });

function localTimestamp() {
    const now = new Date();
    const pad = (value) => String(value).padStart(2, '0');
    return [
        now.getFullYear(),
        pad(now.getMonth() + 1),
        pad(now.getDate()),
    ].join('-') + ' ' + [
        pad(now.getHours()),
        pad(now.getMinutes()),
        pad(now.getSeconds()),
    ].join(':');
}

function responseMessage(type, success, message, extra = {}) {
    return JSON.stringify({
        type,
        success,
        message,
        timestamp: localTimestamp(),
        ...extra,
    });
}

function getUEClients(sender = null) {
    return Array.from(wss.clients).filter(
        (client) => (
            client !== sender
            && client.clientRole !== 'controller'
            && client.readyState === WebSocket.OPEN
        )
    );
}

function sendToUEClients(message, sender = null) {
    const clients = getUEClients(sender);
    for (const client of clients) {
        client.send(message);
    }
    return clients.length;
}

function clientRoleFromRequest(request) {
    const requestPath = typeof request.url === 'string' ? request.url : '';
    const queryStart = requestPath.indexOf('?');
    if (queryStart < 0) {
        return 'ue';
    }
    const parameters = new URLSearchParams(requestPath.slice(queryStart + 1));
    return parameters.get('role') || 'ue';
}

wss.on('connection', (ws, request) => {
    ws.clientRole = clientRoleFromRequest(request);
    console.log(`客户端连接: role=${ws.clientRole}`);

    ws.on('message', (message) => {
        const msg = message.toString();
        console.log('收到消息:', msg);
        
        // 保留现有文本心跳兼容。
        if (msg === 'ping') {
            console.log('回复: pong');
            ws.send('pong');
            return;
        }

        let command;
        try {
            command = JSON.parse(msg);
        } catch (error) {
            ws.send(responseMessage(
                'invalid_json',
                false,
                '消息不是有效的JSON'
            ));
            return;
        }

        if (
            command === null
            || Array.isArray(command)
            || typeof command !== 'object'
        ) {
            ws.send(responseMessage(
                'invalid_request',
                false,
                '请求必须是JSON对象'
            ));
            return;
        }

        if (command.type === 'get_connection_status') {
            const ueClients = getUEClients().length;
            ws.send(responseMessage(
                'get_connection_status',
                true,
                ueClients > 0 ? 'UE已连接' : '等待UE连接',
                { ue_clients: ueClients }
            ));
            return;
        }

        if (
            command.type !== 'set_scene_mode'
            || typeof command.mode !== 'string'
            || command.mode.trim().length === 0
        ) {
            ws.send(responseMessage(
                typeof command.type === 'string'
                    ? command.type
                    : 'unknown_request',
                false,
                '不支持的请求，或mode不是非空字符串'
            ));
            return;
        }

        const forwardedCommand = {
            ...command,
            type: 'set_scene_mode',
            mode: command.mode.trim(),
        };
        const forwardedMessage = JSON.stringify(forwardedCommand);
        const receivers = sendToUEClients(forwardedMessage, ws);
        if (receivers < 1) {
            ws.send(responseMessage(
                'set_scene_mode',
                false,
                'UE尚未连接，场景模式未发送',
                {
                    mode: forwardedCommand.mode,
                    receivers: 0,
                }
            ));
            return;
        }
        ws.send(responseMessage(
            'set_scene_mode',
            true,
            `场景模式已发送给${receivers}个UE客户端`,
            {
                mode: forwardedCommand.mode,
                receivers,
            }
        ));
        console.log(
            `场景模式已更新: ${forwardedCommand.mode}, UE接收端=${receivers}`
        );
    });
    
    ws.on('close', () => console.log(`连接关闭: role=${ws.clientRole}`));
});

wss.on('error', (error) => {
    console.error('WebSocket 服务器错误:', error.message);
});

console.log(`WebSocket 服务器已启动: ws://${host}:${port}`);
