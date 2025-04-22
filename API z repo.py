import os
import time
import requests
import numpy as np
import networkx as nx
from bs4 import BeautifulSoup

import vtk
from vtk.util.numpy_support import vtk_to_numpy

from skimage.morphology import skeletonize_3d
from scipy.interpolate import splprep, splev

import pyvista as pv

# Selenium i ChromeDriverManager
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

chrome_opts = Options()
chrome_opts.add_argument("--headless")
chrome_opts.add_argument("--disable-gpu")
chrome_opts.add_argument("--incognito")
chrome_opts.add_argument("--disable-application-cache")
chrome_opts.add_argument("--disable-extensions")
chrome_opts.add_argument("--disable-dev-shm-usage")
chrome_opts.add_argument("--no-sandbox")

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_opts)
