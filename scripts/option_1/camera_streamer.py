import cv2
import socket
import struct
import time
import threading

def get_local_ips():
    """Returns a list of all IP addresses associated with the local machine."""
    ips = []
    try:
        hostname = socket.gethostname()
        ips = socket.gethostbyname_ex(hostname)[2]
    except Exception:
        pass
    
    # Filter list
    ips = [ip for ip in ips if not ip.startswith("127.")]
    
    # Proactively fetch primary interface IP using UDP socket trick
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        primary_ip = s.getsockname()[0]
        s.close()
        if primary_ip not in ips and primary_ip != "127.0.0.1":
            ips.insert(0, primary_ip)
    except Exception:
        pass

    if not ips:
        ips = ["127.0.0.1"]
    return ips

def client_handler(conn, addr, cap, lock, shutdown_event):
    print(f"\n[CONNECTION] Live connection established with client: {addr[0]}:{addr[1]}")
    conn.settimeout(3.0) # 3-second timeout for socket operations
    
    try:
        while not shutdown_event.is_set():
            # Thread-safe camera read
            with lock:
                ret, frame = cap.read()
            
            if not ret:
                time.sleep(0.03)
                continue
            
            # Compress to JPEG
            # Quality 75 provides a great balance between quality and bandwidth
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
            result, encoded_frame = cv2.imencode('.jpg', frame, encode_param)
            if not result:
                continue
                
            # Convert encoded image to bytes
            data = encoded_frame.tobytes()
            size = len(data)
            
            # Send packet header: 4-byte big-endian unsigned int for frame size
            conn.sendall(struct.pack("!I", size))
            # Send image data
            conn.sendall(data)
            
            # Cap transmission rate around 30 FPS to avoid overloading network buffers
            time.sleep(1.0 / 30.0)
            
    except (socket.error, ConnectionResetError, BrokenPipeError) as e:
        print(f"\n[DISCONNECT] Client {addr[0]}:{addr[1]} disconnected: {e}")
    except Exception as e:
        print(f"\n[ERROR] Exception in connection thread: {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass
        print(f"[STATUS] Connection closed. Awaiting new connections...")

def main():
    print("=========================================================")
    print("          CYBERNETIC CAMERA STREAMING SERVER (CAM.EXE)   ")
    print("=========================================================")
    
    # Discover and display local network configuration
    local_ips = get_local_ips()
    print("\n[INFO] Network Discovery Active.")
    print("Recommended IP address(es) to enter in your Monitoring client:")
    for ip in local_ips:
        print(f"  ->  {ip}")
    
    # Configure Port
    port_input = input("\nEnter stream port [Default: 5000]: ").strip()
    port = 5000
    if port_input.isdigit():
        port = int(port_input)
        
    # Initialize Camera
    print("\n[INFO] Initializing hardware camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[CRITICAL] Could not access camera hardware. Verify it is connected/not in use.")
        input("Press Enter to exit...")
        return
        
    # Configure camera frame size for efficient network transmission
    # 640x480 is standard, fast, and uses minimal bandwidth
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Bind Server Socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind(("0.0.0.0", port))
        server_socket.listen(1)
        server_socket.settimeout(1.0) # 1-second timeout to check for program exit loop
    except Exception as e:
        print(f"[CRITICAL] Failed to bind to port {port}: {e}")
        cap.release()
        input("Press Enter to exit...")
        return
        
    print(f"[STATUS] Server actively listening on PORT {port}...")
    print("Press 'q' in the Preview window or Ctrl+C in console to terminate.")
    
    shutdown_event = threading.Event()
    cam_lock = threading.Lock()
    client_thread = None
    
    # Local Preview Window flag
    show_preview = True
    
    try:
        while True:
            # Handle local preview rendering
            if show_preview:
                with cam_lock:
                    ret, frame = cap.read()
                if ret:
                    # Draw server info on preview frame for clarity
                    cv2.putText(frame, f"Server IP: {local_ips[0]}:{port}", (15, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, "STREAM ACTIVE - Press 'Q' to Exit", (15, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    cv2.imshow("cam.exe - Local Camera Feed", frame)
                
                # Check for exit key in OpenCV Window
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n[EXIT] Terminated by local user action.")
                    break
                    
            # Accept connections
            try:
                conn, addr = server_socket.accept()
                
                # If there's an active connection already, join it before starting a new one
                if client_thread and client_thread.is_alive():
                    shutdown_event.set()
                    client_thread.join()
                    shutdown_event.clear()
                    
                client_thread = threading.Thread(
                    target=client_handler, 
                    args=(conn, addr, cap, cam_lock, shutdown_event),
                    daemon=True
                )
                client_thread.start()
                
            except socket.timeout:
                # Regular timeout to keep main thread active and responsive to interrupts
                continue
                
    except KeyboardInterrupt:
        print("\n[EXIT] Shutdown triggered via console keyboard interrupt.")
    finally:
        print("\n[STATUS] Commencing shutdown protocols...")
        shutdown_event.set()
        if client_thread:
            client_thread.join(timeout=1.0)
            
        try:
            server_socket.close()
        except Exception:
            pass
            
        cap.release()
        cv2.destroyAllWindows()
        print("[STATUS] Offline. System Safe.")
        time.sleep(1)

if __name__ == "__main__":
    main()
