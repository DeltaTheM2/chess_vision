<!-- ABOUT THE PROJECT -->
## About The Project



This project is a chess movement recording system that uses a Raspberry Pi and a camera to track and record chess moves automatically. By utilizing AI-based image recognition with YOLOv8 and PyTorch, the system can detect piece movements on the board and convert them into standard chess notation. This allows players to review their games, analyze strategies, and improve their skills. The system provides an efficient and hands-free way to record chess matches, making it useful for casual players, coaches, and competitive analysis.


<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With


* Pyhton
* pyTorch
* yolov8
* Rassbery Pi 5
* Camera Module

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Installation

for the installation process, make sure you first have a virtual evironment installed and enabled 
```sh
conda create -n chess_vision python=3.11
```
after that use the requirements.txt file in the repository to install all of the dependencies
```
pip3 install -r requirements.txt
```


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

It is used in chess tournaments so players dont have to write their moves down

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Instructions

First, position the raspbery Pi above the chessboard, then use the following commands to run the scripts in a sequence.
```
python run_calibration_chess_board.py
```
```
python run_game_chess_board.py
```