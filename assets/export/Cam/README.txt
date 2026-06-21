TRAFFIC VEHICLE MONITORING SYSTEM — cam.exe
=============================================

This runs on the laptop with the physical camera. No Python installation
required.

HOW TO RUN
  Double-click cam.exe. A console window will open.

SETUP
  1. It will print the IP address(es) of this machine and prompt for a
     port (press Enter to use the default: 5000).
  2. A local preview window opens showing the camera feed.
  3. On the OTHER laptop, open car.exe -> Settings (gear icon) and enter
     this machine's IP address and the port shown here.
  4. Then use car.exe's "Connect Live Camera" option.

NOTES
  - Both laptops must be on the same network (e.g. same Wi-Fi).
  - If the connection fails, check Windows Firewall isn't blocking
    cam.exe or the chosen port (allow it on Private networks).
  - Press 'Q' in the preview window, or Ctrl+C in the console, to stop.
