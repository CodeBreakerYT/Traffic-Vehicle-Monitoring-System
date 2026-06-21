TRAFFIC VEHICLE MONITORING SYSTEM — car.exe
=============================================

This is the main application. No Python installation required.

HOW TO RUN
  Double-click car.exe.

OPTIONS
  1. Connect Live Camera   - streams from a 2nd laptop running cam.exe
                              (see the "Cam" folder). Set its IP/port via
                              the gear icon (Settings) on the main menu.
  2. Process Video / GIF    - upload a traffic video file to analyse for
                              vehicles, accidents, and severity.
  3. CARLA Simulation       - requires CARLA installed separately. If not
                              installed, a "DOWNLOAD CARLA" button appears
                              on the options screen.

SETTINGS
  Click the gear icon (top-right of main menu) to configure:
    - CARLA installation path
    - Camera IP / port (for the live camera option)

NOTES
  - config.json (in this folder) stores your settings. Safe to delete —
    it will be recreated with defaults on next launch.
  - Recordings made in-app are saved to a "recordings" folder created
    next to car.exe.
