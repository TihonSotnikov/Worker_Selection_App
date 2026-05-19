import socket
import os

STUN_SERVERS = [
    ("stun.nextcloud.com", 443),
    # ("stun.l.google.com", 19302), # Не работает
    ("stun.cloudflare.com", 3478),
    ("stun.sipnet.ru", 3478),
    # ("stun.ekiga.net", 3478), # Не работает
]

def parse_stun_response(data, sent_tx_id):
    if len(data) < 20:
        return "Пакет слишком короткий"
    
    # Заголовок (20 байт)
    msg_type = data[0:2]
    msg_len = int.from_bytes(data[2:4], 'big')
    magic_cookie = data[4:8]  # Ожидается b"\x21\x12\xA4\x42"
    resp_tx_id = data[8:20]
    
    if resp_tx_id != sent_tx_id:
        return "Transaction ID не совпадает"
        
    # Парсим атрибуты (начинаются с 20-го байта)
    offset = 20
    end_offset = 20 + msg_len
    
    ip_addr = None
    port = None
    
    while offset < end_offset:
        if offset + 4 > len(data):
            break
            
        attr_type = int.from_bytes(data[offset:offset+2], 'big')
        attr_len = int.from_bytes(data[offset+2:offset+4], 'big')
        
        offset += 4
        if offset + attr_len > len(data):
            break
            
        attr_val = data[offset : offset + attr_len]
        
        # Переходим к следующему атрибуту с учетом выравнивания по 4 байта
        padded_len = (attr_len + 3) & ~3
        offset += padded_len
        
        # XOR-MAPPED-ADDRESS (Атрибут 0x0020)
        if attr_type == 0x0020:
            family = attr_val[1] # 1 = IPv4, 2 = IPv6
            x_port = int.from_bytes(attr_val[2:4], 'big')
            
            # Дешифруем порт: XOR-им с первыми 16 битами Magic Cookie (0x2112)
            port = x_port ^ 0x2112
            
            if family == 1:  # IPv4 (длина 4 байта)
                x_ip = attr_val[4:8]
                # Дешифруем IP: XOR-им с полным Magic Cookie (0x2112A442)
                ip_bytes = bytes(a ^ b for a, b in zip(x_ip, b"\x21\x12\xA4\x42"))
                ip_addr = socket.inet_ntoa(ip_bytes)
            elif family == 2:  # IPv6 (длина 16 байт)
                x_ip = attr_val[4:20]
                # Дешифруем IP: XOR-им с Magic Cookie + Transaction ID
                xor_key = b"\x21\x12\xA4\x42" + sent_tx_id
                ip_bytes = bytes(a ^ b for a, b in zip(x_ip, xor_key))
                ip_addr = socket.inet_ntop(socket.AF_INET6, ip_bytes)
                
        # Классический MAPPED-ADDRESS (Атрибут 0x0001) - без XOR шифрования
        elif attr_type == 0x0001:
            family = attr_val[1]
            port = int.from_bytes(attr_val[2:4], 'big')
            if family == 1:
                ip_addr = socket.inet_ntoa(attr_val[4:8])
            elif family == 2:
                ip_addr = socket.inet_ntop(socket.AF_INET6, attr_val[4:20])

    return ip_addr, port

def test_stun():
    for target in STUN_SERVERS:
        tx_id = os.urandom(12)
        # RFC 5389 заголовок: Запрос (0x0001) + Длина (0x0000) + Magic Cookie + Transaction ID
        message = b"\x00\x01\x00\x00\x21\x12\xA4\x42" + tx_id
        
        # Для корректной резолвации доменного имени принудительно берем только IPv4
        try:
            addr_info = socket.getaddrinfo(target[0], target[1], socket.AF_INET, socket.SOCK_DGRAM)
            resolved_target = addr_info[0][4]
        except Exception as dns_err:
            print(f"Ошибка DNS для {target[0]}: {dns_err}")
            continue
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(3)
        
        try:
            print(f"\nОтправка на {target[0]}:{target[1]} (IP: {resolved_target[0]})...")
            sock.sendto(message, resolved_target)
            data, addr = sock.recvfrom(2048)
            
            result = parse_stun_response(data, tx_id)
            if isinstance(result, tuple):
                print(f"Успех! Ваш публичный адрес: {result[0]}:{result[1]}")
            else:
                print(f"Ответ получен, но возникла ошибка парсинга: {result}")
                
        except socket.timeout:
            print("Тайм-аут: ответа нет.")
        except Exception as e:
            print(f"Ошибка: {e}")
        finally:
            sock.close()

if __name__ == "__main__":
    test_stun()
