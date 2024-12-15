Hello!
If you're reading this you're probably trying to understand Carla. Well here are some steps to installing Carla, and using the Simulator.

# Installation Requirements / Steps
What to get
- Python 3.10
- Carla 9.14.0 Package from Github
- 8GB VRAM

1. In integrated terminal run: 
    - pip install --user pygame numpy
2. Extract Carla 9.14.0 to a folder
3. Add the .whl file to Carla\PythonAPI\Carla\dist
    - pip install <wheel-file-name>.whl
4. Return to root of package 
    -  CarlaUE4.exe
5. Ensure Carla does not crash :/

# Running Carla
1. Navigate to file location
    cd C:\Users\django\Documents\Alex\WindowsNoEditor
2. Run in terminal: 
    .\CarlaUE4 -quality-level=Low