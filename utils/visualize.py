import torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import io
import PIL.Image
from torchvision.transforms import ToTensor
import numpy as np
import glob
import os
import random
import matplotlib.animation as animation
""" This code contains visualization functions to generate GIFs. Code is messy, use code at your own risk"""


def make_gif(y_softmax=None,
			 y=None,
			 global_patch=None,
			 grid_size_in_global=None,
			 probability_mask=None,
			 return_mode="buf",
			 axs=None,
			 input_trajectory=None,
			 gt_trajectory=None,
			 prediction_trajectories=None,
			 background_image=None,
			 img_scaling=None,
			 final_position = None,
			 scaling_global=None,
			 num_traj = 1

			 ):

	def set_axis(ax, legend_elements):
		plt.axis('off')
		ax.axes.get_xaxis().set_visible(False)
		ax.axes.get_yaxis().set_visible(False)

		ax.set_xlim(extent[0], extent[1])
		ax.set_ylim(extent[3], extent[2])
		legend = ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.00), ncol = 3, framealpha=0, columnspacing =1, handletextpad= 0.4)

		plt.setp(legend.get_texts(), color='w')
		buf = io.BytesIO()

		plt.savefig(buf, bbox_inches='tight', format='jpeg', dpi=300)
		plt.close()
		buf.seek(0)
		image = PIL.Image.open(buf)

		return image
	from PIL import Image, ImageDraw
	from matplotlib.lines import Line2D



	if img_scaling:
		if not type(input_trajectory) == type(None): input_trajectory /= img_scaling
		if not type(prediction_trajectories) == type(None): prediction_trajectories /= img_scaling
		if not type(gt_trajectory) == type(None): gt_trajectory /= img_scaling
		if not type(final_position) == type(None): final_position /= img_scaling
	images = []

	color = "red"
	color_goal = "red"
	marker_input = "o"
	marker_output = "v"
	marker_goal = "*"
	marker_size = 3



	legend_elements = [Line2D([0], [0], marker = marker_input ,color='w', lw=0 ,  markerfacecolor = color ,markeredgecolor=color,  label="Input") ,
					   Line2D([0], [0], marker=marker_output,color='w', lw=0 , markerfacecolor=color, markeredgecolor=color, label="Prediction"),
					   Line2D([0], [0], marker=marker_goal,color='w', lw=0 , markerfacecolor=color_goal, markeredgecolor=color_goal, label="Goal")]



	center = input_trajectory[-1, 0]
	dx = (grid_size_in_global + 0.5) / img_scaling


	extent = [center[0] - dx, center[0] + dx \
		, center[1] + dx, center[1] - dx]

	if any(np.array(extent) < 0):
		return 0

	final_position += center.view(1, 1, -1)


	max_dist = 0
	for j in range(1, final_position.size(1)):
		dist = torch.norm( final_position[0, 0]-final_position[0, j], 2)

		if dist > max_dist:
			max_dist = dist
			index = j

	trajectories_id = [0, index]

	for j in range(1, final_position.size(1)):
		if j != index:
			trajectories_id.append(j)
		if len(trajectories_id) == num_traj:
			break


	for num  in trajectories_id:

		for i in range(2, len(input_trajectory)):
			fig, ax = plt.subplots()

			plt.imshow(background_image)
			for k in range(3):
				plt.plot(input_trajectory[i - k, num , 0], input_trajectory[i - k, num , 1], marker=marker_input, color=color,
						 markersize=marker_size, label="Input")

			images.append(set_axis(ax, legend_elements))

		softmax_img = np.clip(y_softmax[num][0 ].cpu().numpy(), 0, 1)
		softmax_img /= np.max(softmax_img)
		softmax_img = Image.fromarray(softmax_img)
		softmax_img = ToTensor()(np.array(
			softmax_img.resize(
				(int(2*dx), int(2 * dx)), resample = PIL.Image.LANCZOS)))


		color_map = (torch.tensor([1., 1., 0.]).view(-1, 1, 1) - torch.ones(3, int(2*dx), int(2 * dx)) * softmax_img  * torch.tensor([0., 1., 0.]).view(-1, 1, 1) )
		softmax_img = torch.cat((color_map, softmax_img), dim=0).permute(1, 2, 0)
		final_softmax_image = torch.zeros(background_image.width, background_image.height, 4)

		final_softmax_image[int(extent[3]) :int(extent[3]) +int(2*dx), int(extent[0]):int(extent[0])+ int(2*dx) ] = softmax_img
		time_duration = 5


		# print(softmax_img.size() )


		## Show probability
		for t in range(time_duration):
			fig, ax = plt.subplots()
			ax.imshow(background_image)
			ax.imshow(final_softmax_image)

			for k in range(3):
				plt.plot(input_trajectory[i - k, num , 0], input_trajectory[i - k, num , 1], marker=marker_input, color=color,
						 markersize=marker_size)



			images.append(set_axis(ax, legend_elements))

		## Show probability
		for t in range(time_duration):
			fig, ax = plt.subplots()
			ax.imshow(background_image)
			ax.imshow(final_softmax_image, interpolation = "lanczos")

			plt.plot(final_position[0, num , 0], final_position[0, num , 1], marker=marker_goal, color=color_goal, label="Goal ")
			for k in range(3):
				plt.plot(input_trajectory[i - k, num , 0], input_trajectory[i - k, num , 1], marker=marker_input, color=color,
						 markersize=marker_size)

			images.append(set_axis(ax, legend_elements))

		for t in range(time_duration):
			fig, ax = plt.subplots()
			ax.imshow(background_image)
			ax.imshow(final_softmax_image, interpolation = "lanczos")


			plt.plot(final_position[0, num , 0], final_position[0, num, 1], marker=marker_goal, color=color_goal)
			for k in range(3):
				plt.plot(input_trajectory[i - k, num , 0], input_trajectory[i - k, num, 1], marker=marker_input, color=color,
						 markersize=marker_size)

			images.append(set_axis(ax, legend_elements))
		# PREDICTION
		for i in range(0, len(prediction_trajectories)):
			fig, ax = plt.subplots()
			ax.imshow(background_image)
			plt.plot(final_position[0, num , 0], final_position[0, num , 1], marker=marker_goal, color=color_goal)
			for k in range(3):

				ind = i - k
				if ind < 0:
					plt.plot(input_trajectory[ind, num , 0], input_trajectory[ind, num , 1], marker=marker_output, color=color,
							 markersize=marker_size)
				else:
					plt.plot(prediction_trajectories[ind, num , 0], prediction_trajectories[ind, num , 1], marker=marker_output,
							 color=color, markersize=marker_size)

			images.append(set_axis(ax, legend_elements))

	GIF_DIRECTORY = "images/gif"
	gifs = glob.glob(os.path.join(GIF_DIRECTORY, "*.gif"))
	import re
	if len(gifs) < 1:
		number = [-1]
	else:
		number = [int(re.search(r'\d+', file).group()) for file in  gifs]
	# from IPython.display import display
	# display(images[0])
	# display(images[10])
	# print(images)
	# images[0].save('images/gif/gif_{}'.format(max(number)+1), "JPEG", quality=10  )
	#
	# fdfh
	#

	images[0].save('images/gif/gif_{}.gif'.format(max(number)+1),
				   save_all=True, append_images=images[1:], duration=200, loop=0)

	print("saved")
def visualize_probabilities(y_softmax = None,
							y = None,
							global_patch = None,
							grid_size_in_global = None,
							probability_mask = None,
							return_mode = "buf",
							axs = None):

	""""""
	color = torch.ones(3, int(2 * grid_size_in_global + 1),
					   int(2 * grid_size_in_global + 1)) * torch.tensor([1., 0., 0.]).view(-1, 1, 1)
	recon_img = np.clip(y[0][0].cpu().numpy(), 0, 1)
	recon_img = Image.fromarray(recon_img)
	recon_img = ToTensor()(np.array(recon_img.resize(
		(int(2 * grid_size_in_global + 1), int(2 * grid_size_in_global + 1)))))
	recon_img = torch.cat((color, recon_img.cpu()), dim=0).permute(1, 2, 0)

	softmax_img = np.clip(y_softmax[0][0].cpu().numpy(), 0, 1)
	softmax_img /= np.max(softmax_img)
	softmax_img = Image.fromarray(softmax_img)
	softmax_img = ToTensor()(np.array(
		softmax_img.resize(
			(int(2 * grid_size_in_global + 1), int(2 * grid_size_in_global + 1)))))

	softmax_img = torch.cat((color, softmax_img.cpu()), dim=0).permute(1, 2, 0)

	gt_img = np.clip(probability_mask, 0, 1)
	gt_img = Image.fromarray(gt_img)
	gt_img = ToTensor()(np.array(gt_img.resize(
		(int(2 * grid_size_in_global + 1), int(2 * grid_size_in_global + 1)))))
	gt_img = torch.cat((color, gt_img.cpu()), dim=0).permute(1, 2, 0)

	real_img = np.clip(global_patch, 0, 1)

	if axs is None:
		fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)

	axs[0].imshow(np.transpose(real_img, (1, 2, 0)), interpolation='nearest')#,  origin='upper')
	axs[0].imshow(softmax_img, interpolation='nearest')
	axs[0].set_title("Probability Map")
	axs[0].axis('off')

	axs[1].imshow(np.transpose(real_img, (1, 2, 0)), interpolation='nearest')#,  origin='upper')
	axs[1].imshow(recon_img, interpolation='nearest', )
	axs[1].set_title("Goal Realisation")
	axs[1].axis('off')

	axs[2].imshow(np.transpose(real_img, (1, 2, 0)), interpolation='nearest')#,  origin='upper')
	axs[2].imshow(gt_img, interpolation='nearest')
	axs[2].set_title("GT Goal")
	axs[2].axis('off')

	if return_mode == "buf":
		buf = io.BytesIO()
		plt.savefig(buf, format='jpeg')
		plt.close()
		buf.seek(0)
		image = PIL.Image.open(buf)
		image = ToTensor()(image)
		return image
	elif return_mode == "ax":
		return axs
	else:
		return fig

def visualize_trajectories(input_trajectory = None,
						   gt_trajectory = None,
						   prediction_trajectories = None,
						   background_image = None,
						   img_scaling = None,
						   scaling_global = None,
						   grid_size = 20 ,
						   return_mode = "buf",
						   axs = None):



	# re-scale trajectories
	if img_scaling:
		if not type(input_trajectory) == type(None): input_trajectory/=img_scaling
		if not type(prediction_trajectories) == type(None): prediction_trajectories/=img_scaling
		if not type(gt_trajectory) == type(None): gt_trajectory/= img_scaling

	center = input_trajectory[-1]
	dx = (grid_size + 0.5)/img_scaling*scaling_global
	if axs is None:
		fig_trajectory, axs = plt.subplots()

	axs.imshow(background_image,  alpha=0.9)

	extent = [center[0] - dx, center[0] + dx \
		, center[1] + dx, center[1] - dx]

	axs.scatter(input_trajectory[:, 0], input_trajectory[:, 1], color="black", marker="o", s=2)
	marker_style = dict(color='tab:orange', marker='o',
						markersize=3, markerfacecoloralt='tab:red', markeredgewidth=1)

	for i in range(prediction_trajectories.size(1)):
	 	#ax_trajectory.plot(prediction_trajectories[:, i, 0], prediction_trajectories[:, i,1], linestyle='None', fillstyle="none", **marker_style)
		axs.plot(prediction_trajectories[:, i, 0], prediction_trajectories[:, i, 1], color="red")

	axs.axis('off')
	#
	# axs.set_xlim(extent[0] - grid_size, extent[1] + grid_size)
	# axs.set_ylim(extent[3] - grid_size, extent[2] + grid_size)

	if return_mode == "buf":
		buf = io.BytesIO()
		plt.savefig(buf, format='jpeg')
		plt.close()
		buf.seek(0)
		image = PIL.Image.open(buf)
		image = ToTensor()(image)
		return image
	elif return_mode=="ax":
		return axs
	else:
		return fig_trajectory


def visualize_traj_probabilities(input_trajectory=None,
							   gt_trajectory=None,
							   prediction_trajectories=None,
							   background_image=None,
							   img_scaling=None,
							   scaling_global=None,
							   grid_size=20,
								y_softmax = None,
								 y=None,
								 global_patch=None,
								 grid_size_in_global=None,
								 probability_mask=None,
								 buf="buf"):
	fig, axs = plt.subplots(1, 4, figsize=(16, 4))


	visualize_probabilities(y_softmax = y_softmax,
							y = y,
							global_patch = global_patch,
							grid_size_in_global = grid_size_in_global,
							probability_mask = probability_mask,
							return_mode = "ax",
							axs = axs[1:])

	visualize_trajectories(input_trajectory = input_trajectory,
							   gt_trajectory = gt_trajectory,
							   prediction_trajectories = prediction_trajectories,
							   background_image = background_image,
							   img_scaling = img_scaling,
							   scaling_global = scaling_global,
							   grid_size = 20,
								return_mode = "ax",
								axs=axs[0])




	if buf:
		buf = io.BytesIO()
		plt.savefig(buf, format='jpeg')
		plt.close()
		buf.seek(0)
		image = PIL.Image.open(buf)
		image = ToTensor()(image)
		return image
	else:
		return fig_trajectory





def make_gif_dataset(dataset, k =20):

	def set_axis(ax, legend_elements):
		plt.axis('off')
		ax.axes.get_xaxis().set_visible(False)
		ax.axes.get_yaxis().set_visible(False)
		legend = ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.00), ncol=3,
						   framealpha=0, columnspacing=1, handletextpad=0.4)

		plt.setp(legend.get_texts(), color='w')
		# buf = io.BytesIO()
		#
		# plt.savefig(buf, bbox_inches='tight', format='jpeg')
		# plt.close()
		# buf.seek(0)
		# image = Image.open(buf).convert('P')

		return ax
	from PIL import Image, ImageDraw
	from matplotlib.lines import Line2D

	# Tableau 20 Colors
	tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
				 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
				 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
				 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
				 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]



	for i in range(len(tableau20)):
		r, g, b = tableau20[i]
		tableau20[i] = (r / 255., g / 255., b / 255.)
	random.shuffle(tableau20)
	img_scaling = dataset.img_scaling


	marker_input = "o"
	marker_output = "v"


	range_list = list(range(200))
	random.shuffle(range_list)
	fig = plt.figure()
	img_id = 0
	for c_id, index in enumerate(range_list[:k]):
		item = dataset.__getitem__(index)
		print(index)
		input_trajectory = item[0]
		gt_trajectory = item[1]
		if img_scaling:
			if not type(input_trajectory) == type(None): input_trajectory /= img_scaling
			if not type(gt_trajectory) == type(None): gt_trajectory /= img_scaling

		legend_elements = [
			Line2D([0], [0], marker=marker_input, color='w', lw=0, markerfacecolor=tableau20[c_id], markeredgecolor=tableau20[c_id],
				   label="Input"),
			Line2D([0], [0], marker=marker_output, color='w', lw=0, markerfacecolor=tableau20[c_id], markeredgecolor=tableau20[c_id],
				   label="Ground-truth")]

		for traj in input_trajectory[0]:
			f , ax = plt.subplots()
			ax.imshow(item[4][0]["scaled_image"])
			im = ax.plot(traj[0], traj[1], marker=marker_input, color = tableau20[c_id])
			set_axis(ax, legend_elements)
			# plt.show()
			plt.tight_layout()
			plt.savefig("images/gif_dataset/{}.jpg".format(img_id), bbox_inches='tight', pad_inches=0,  dpi = 300)
			
			img_id += 1
			plt.close()

		for traj in gt_trajectory[0]:
			f , ax = plt.subplots()
			ax.imshow(item[4][0]["scaled_image"])
			plt.plot(traj[0], traj[1], marker=marker_output, color = tableau20[c_id])
			set_axis(ax, legend_elements)
			# plt.show()
			plt.tight_layout()
			plt.savefig("images/gif_dataset/{}.jpg".format(img_id), bbox_inches='tight', pad_inches=0, dpi=300)

			img_id+=1
			plt.close()
	