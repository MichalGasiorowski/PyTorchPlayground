import gym
import torch
import numpy as np
import torchvision.transforms as T
import matplotlib.pyplot as plt
from IPython import display as ipythondisplay

class CartPoleEnvManager():
	def __init__(self, device, env_wrapper=lambda x: x, timestep_limit = 100, xvfb_mode=False):
		self.device = device
		#env = gym.make('CartPole-v0').unwrapped
		env = gym.make('CartPole-v0')
		env.spec.timestep_limit = timestep_limit
		self.env = env_wrapper(env)
		#self.env = env
		self.env.reset()
		self.current_screen = None
		self.done = False
		self.xvfb_mode = xvfb_mode

	def reset(self):
		self.env.reset()
		self.current_screen = None

	def close(self):
		self.env.close()

	def render(self, mode='human'):
		screen = self.env.render(mode)
		return screen

	def num_actions_available(self):
		return self.env.action_space.n

	def take_action(self, action):
		#print('in take_action; done=', self.env.env.done)
		_, reward, self.done, _ = self.env.step(action.item())
		return torch.tensor([reward], device=self.device) # step() expects normal number, not torch tensor!

	def just_starting(self):
		return self.current_screen is None

	# The state is defined as the difference between the current screen and the previous screen
	def get_state(self):
		if self.just_starting() or self.done:
			self.current_screen = self.get_processed_screen()
			black_screen = torch.zeros_like(self.current_screen)
			return black_screen
		else:
			s1 = self.current_screen
			s2 = self.get_processed_screen()
			self.current_screen = s2
			return s2 - s1

	def get_screen_height(self):
		screen = self.get_processed_screen()
		return screen.shape[2]

	def get_screen_width(self):
		screen = self.get_processed_screen()
		return screen.shape[3]

	def get_processed_screen(self):
		screen = self.render('rgb_array')
		screen = self.crop_screen(screen, hwc=True)
		if self.xvfb_mode:
			plt.imshow(screen)
			#ipythondisplay.clear_output(wait=True)
			ipythondisplay.display(plt.gcf())
		return self.transform_screen_data(screen.transpose((2, 0, 1))) # CHW is expected HWC -> CHW

	def crop_screen(self, screen, hwc=True):
		screen_height = screen.shape[0] if hwc else screen.shape[1]
		screen_width = screen.shape[1] if hwc else screen.shape[0]
		# Strip off top bottom
		top = int(screen_height * 0.4)
		bottom = int(screen_height * 0.8)
		left = int(screen_width * 0.2)
		right = int(screen_width * 0.8)
		screen = screen[top:bottom, left:right, :] if hwc else screen[left:right, top:bottom, :]
		return screen

	def transform_screen_data(self, screen):
		# Convert to float, rescale, convert to tensor
		screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
		screen = torch.from_numpy(screen)

		# use torchvision package to compose image transformations
		resize = T.Compose([
			T.ToPILImage()
			,T.Resize((40,90))
			,T.ToTensor()
		])

		return resize(screen).unsqueeze(0).to(self.device)
