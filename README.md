<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Best-README-Template</h3>

  <p align="center">
    An awesome README template to jumpstart your projects!
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template">View Demo</a>
    &middot;
    <a href="https://github.com/othneildrew/Best-README-Template/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/othneildrew/Best-README-Template/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



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