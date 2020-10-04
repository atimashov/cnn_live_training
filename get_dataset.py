from bs4 import BeautifulSoup
from argparse import  ArgumentParser
import numpy as np
import requests
import cv2
import urllib
from datetime import datetime as dt
import os

classes_id = {
	'king_penguin': 'n02056570', # 2022
	'giant_panda': 'n02510455', # 1832
	'red_panda': 'n02509815', # 1686
	'wombat': 'n01877812', # 1222
	'echidna': 'n01872401', # 1336
	'llama': 'n02437616', # 1304
	'hippo': 'n02422699', # 1391
	'alaskan_malamute': 'n02110063', # 1634
	'baboon': 'n02486410', # 1635
	'otter': 'n02444819', # 1547
}

def get_class_urls(class_id):
	page = requests.get("http://www.image-net.org/api/text/imagenet.synset.geturls?wnid={}".format(class_id))
	soup = BeautifulSoup(page.content, 'html.parser')
	str_soup = str(soup)
	return str_soup.split('\r\n')

def url_to_image(url):
    resp = urllib.request.urlopen(url, timeout = 1)
    image = np.asarray(bytearray(resp.read()), dtype = 'uint8')
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def save_images(urls, path):
	i = 0
	for j, url in enumerate(urls):
		try: I = url_to_image(url)
		except: continue
		if I is None: continue
		save_path = '{}/img_{}.jpg'.format(path, i)
		cv2.imwrite(save_path, I)
		i += 1
	return i, j + 1

if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('--dataset-path', type=str, default='data', help='path to dataset, by default "data"')
	parser.add_argument('--print-stat', type=str, default='True', help='should I print statistics of downloading or not')
	inputs = parser.parse_args()
	inputs.print_stat = inputs.print_stat == 'True'
	print(inputs)
	print()
	print('-' * 50)

	s = 0
	if not os.path.exists(inputs.dataset_path): os.makedirs(inputs.dataset_path)
	for animal in classes_id:
		if inputs.print_stat: print('{} started: {}'.format(animal, str(dt.now().time())[:8]))

		# create folder
		folder_path = '{}/{}'.format(inputs.dataset_path, animal)
		if not os.path.exists(folder_path): os.makedirs(folder_path)

		# get the lists of URLs for the images of the synset
		urls = get_class_urls(classes_id[animal])

		# save images
		i, j = save_images(urls, folder_path)
		s += i
		if inputs.print_stat:
			print('{} out of {}'.format(i, j))
			print('{} finished: {}'.format(animal, str(dt.now().time())[:8]))
			print()
	if inputs.print_stat:
		print('-' * 50)
		print('Downloaded {} images'.format(s))