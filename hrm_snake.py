import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pygame
from collections import deque
from snake_game import SnakeEnv
import imageio
import matplotlib.pyplot as plt
import time
import hrm_for_snake
from hrm_for_snake import internal_width

# Hyperparameters
GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = hrm_for_snake.batch_size
MEMORY_SIZE = int(1e6) #10000
EPSILON_START = 0.0
EPSILON_END = 0.0
EPSILON_DECAY = 0.5
TARGET_UPDATE_FREQ = 10
EPISODES = 50000000000000
# THRESHOLD = 0.01
CURIOSITY_SCALING_FACTOR = 10
CURIOSITY_ENABLED = False

RENDER_EVERY = 1 #20

device = torch.device("cuda")
print(f"Using device: {device}\n")

class ReplayMemory:
	def __init__(self, capacity):
		self.buffer = deque(maxlen=capacity)

	def push(self, transition):
		# transition = (state, action, reward, next_state, done)
		self.buffer.append(transition)

	def sample(self, batch_size):
		return random.sample(self.buffer, batch_size)
		
	def __len__(self):
		return len(self.buffer)
	
def train(model_path=None):

	start_time = time.time()

	env = SnakeEnv()
	grid_h, grid_w = env.grid_height, env.grid_width
	n_actions = 4  # right, left, up, down

	policy_net = hrm_for_snake.HRM().to(device)
	target_net = hrm_for_snake.HRM().to(device)
	future_net = hrm_for_snake.HRM(output_size=20*20*4).to(device)

	# policy_net.compile()
	# target_net.compile()
	# future_net.compile()

	print("policy_net:", sum(p.numel() for p in policy_net.parameters()))
	print("target_net:", sum(p.numel() for p in policy_net.parameters()))
	print("future_net:", sum(p.numel() for p in policy_net.parameters()))

	if model_path is not None:
		print(f"Loading model from: {model_path}")
		policy_net.load_state_dict(torch.load(model_path, map_location=device))

	target_net.load_state_dict(policy_net.state_dict())
	target_net.eval()

	optimizer = optim.AdamW(policy_net.parameters(), lr=LR, betas=(0.95, 0.95))
	future_net_optimizer = optim.AdamW(future_net.parameters(), lr=LR, betas=(0.95, 0.95))

	memory = ReplayMemory(MEMORY_SIZE)
	future_memory = ReplayMemory(MEMORY_SIZE)

	epsilon = 0 if model_path is not None else EPSILON_START # can replace this line with epsilon = EPSILON_START to continue training a pre-trained model rather than testing with eps = 0

	best_rendered_score = -float('inf')
	best_score = -float('inf')
	video_filename = 'best_snake_episode.mp4'
	model_save_path = "best_dqn_model.pth"

	all_scores = []
	all_losses = []

	for episode in range(EPISODES):
		obs = env.reset()

		hrm_for_snake.z_init_policy = torch.randn((1, internal_width)), torch.randn((1, internal_width)).to(device)
		hrm_for_snake.z_init_target = torch.randn((1, internal_width)), torch.randn((1, internal_width)).to(device)
		hrm_for_snake.z_init_future = torch.randn((1, internal_width)), torch.randn((1, internal_width)).to(device)


		total_reward = 0
		total_curiosity_reward = 0
		done = False

		render = (episode % RENDER_EVERY == 0)
		# render = False
		frames = []
		episode_losses = []

		# while not done:
		for _ in range(1000):

			if done:
				break

			if render:
				pygame.event.pump()
				env.window.fill('black')
				env.draw_snake()
				env.draw_food()
				env.display_score()
				pygame.display.flip()
				env.clock.tick(30)

				frame = pygame.surfarray.array3d(env.window)
				frame = frame.transpose([1, 0, 2]) # transpose (Width, Height, Channel) to (Height, Width, Channel)
				frames.append(frame) 

			# choose between random action and policy net action
			if random.random() < epsilon:
				with torch.no_grad():
					obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
					hrm_for_snake.z_init_policy, action = policy_net(hrm_for_snake.z_init_policy, obs_tensor)
				action = random.randint(0, n_actions - 1)
			else:
				with torch.no_grad():
					obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
					hrm_for_snake.z_init_policy, action = policy_net(hrm_for_snake.z_init_policy, obs_tensor)
					# print(action.shape)
					action = action.mean(0).argmax().item()
			
			# Step through environment
			# print(action)
			assert action < 4
			next_obs, reward, done = env.step(action)
			if CURIOSITY_ENABLED:
				next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
				hrm_for_snake.z_init_future, next_obs_tensor_p = future_net(hrm_for_snake.z_init_future, obs_tensor)
				curiosity = torch.nn.functional.mse_loss(next_obs_tensor_p, next_obs_tensor.reshape((1, 1, 1600)))
				reward += curiosity.detach().item() * CURIOSITY_SCALING_FACTOR
			n = 0

			memory.push((obs, action, reward, next_obs, done))
			future_memory.push((obs, next_obs))
			obs = next_obs
			total_reward += reward
			if CURIOSITY_ENABLED:
				total_curiosity_reward += curiosity.detach().item() * CURIOSITY_SCALING_FACTOR

			# Train the network
			if len(memory) >= BATCH_SIZE:
				batch = memory.sample(BATCH_SIZE)
				obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*batch)

				# Convert to tensors:
				obs_batch = torch.tensor(np.stack(obs_batch), dtype=torch.float32, device=device)
				action_batch = torch.tensor(action_batch, dtype=torch.int64, device=device).unsqueeze(1)
				reward_batch = torch.tensor(reward_batch, dtype=torch.float32, device=device).unsqueeze(1)
				next_obs_batch = torch.tensor(np.stack(next_obs_batch), dtype=torch.float32, device=device)
				done_batch = torch.tensor(done_batch, dtype=torch.float32, device=device).unsqueeze(1)

				N_supervision = 1

				for _ in range(N_supervision):

					#1 Forward pass
					# print(obs_batch.shape, action_batch.shape)
					# hrm_for_snake.z_init_policy, q_values = policy_net((hrm_for_snake.z_init_policy[0],
						# hrm_for_snake.z_init_policy[1]), obs_batch)
					_, q_values = policy_net((hrm_for_snake.z_init_policy[0],
						hrm_for_snake.z_init_policy[1]), obs_batch)
					# print(q_values.shape)
					q_values = q_values.gather(1, action_batch)
					with torch.no_grad():
						# hrm_for_snake.z_init_policy, max_next_q = target_net((hrm_for_snake.z_init_policy[0],
							# hrm_for_snake.z_init_policy[1]), next_obs_batch)
						_, max_next_q = target_net((hrm_for_snake.z_init_policy[0],
							hrm_for_snake.z_init_policy[1]), next_obs_batch)
						# print(max_next_q.shape)
						# quit()
						max_next_q = torch.max(max_next_q, dim=1, keepdim=True)[0]
						# print(max_next_q.shape)
						target_q = reward_batch + GAMMA * max_next_q * (1 - done_batch)

					#2 Calculate loss
					# loss = nn.MSELoss()(q_values, target_q)
					# print(q_values.shape, target_q.shape)
					loss = nn.functional.mse_loss(q_values, target_q)

					#3 Optimizer zero grad zeros out gradients since gradients can accumulate into next iteration even though they have already been applied 
					optimizer.zero_grad()

					#4 Backprop
					hrm_for_snake.z_init_policy[0].detach()
					hrm_for_snake.z_init_policy[1].detach()
					hrm_for_snake.z_init_target[0].detach()
					hrm_for_snake.z_init_target[1].detach()
					# with torch.autograd.set_detect_anomaly(True):
					loss.backward()
					torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)

					#5 Gradient Descent
					optimizer.step()

					# visualization
					episode_losses.append(loss.item())

					n = 0

				if CURIOSITY_ENABLED:

					future_batch = future_memory.sample(BATCH_SIZE)
					obs_batch, next_obs_batch = zip(*future_batch)
					obs_batch = torch.tensor(np.stack(obs_batch), dtype=torch.float32, device=device)
					next_obs_batch = torch.tensor(np.stack(next_obs_batch), dtype=torch.float32, device=device)
					obs_tensor = obs_batch
					next_obs_tensor = next_obs_batch

					for _ in range(N_supervision):

						future_net_optimizer.zero_grad()
						hrm_for_snake.z_init_future, next_obs_tensor_p = future_net((hrm_for_snake.z_init_future[0],
							hrm_for_snake.z_init_future[1]), obs_tensor)
						curiosity = torch.nn.functional.mse_loss(next_obs_tensor_p, next_obs_tensor.reshape((BATCH_SIZE, 1600)))
						hrm_for_snake.z_init_future[0].detach()
						hrm_for_snake.z_init_future[1].detach()
						curiosity.backward()
						torch.nn.utils.clip_grad_norm_(future_net.parameters(), max_norm=1.0)
						future_net_optimizer.step()
			
		score = env.snake_size - env.STARTING_SIZE

		# visualization
		avg_loss = np.mean(episode_losses) if episode_losses else 0
		all_losses.append(avg_loss)
		all_scores.append(score)

		# Update target network
		if episode % TARGET_UPDATE_FREQ == 0:
			target_net.load_state_dict(policy_net.state_dict())
			hrm_for_snake.z_init_target = hrm_for_snake.z_init_policy

		epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
		print(f"Episode {episode}, Total reward: {round(total_reward, 2)}, Total curiosity reward: {round(total_curiosity_reward, 2)}, Score: {score}, Epsilon: {epsilon:.3f}")

		# NOTE: only saves best score episode AMONG the few ones recorded...
		if render and score > best_rendered_score:
			best_rendered_score = score
			print(f"New best score {best_rendered_score} at episode {episode}. Saving video...")
			imageio.mimwrite(video_filename, frames, fps=15, codec='libx264', quality=8)
		
		if score > best_score: # best total score among all episodes
			best_score = score
			print(f"New best model score: {best_score}. Saving model weights...")
			torch.save(policy_net.state_dict(), model_save_path)

	
	# After training complete:
	window = 50
	episodes = np.arange(len(all_scores))
	rolling_scores = np.convolve(all_scores, np.ones(window)/window, mode='valid')
	rolling_losses = np.convolve(all_losses, np.ones(window)/window, mode='valid')

	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

	ax1.plot(episodes[len(episodes) - len(rolling_scores):], rolling_scores, label=f"Avg Score ({window}-episode window)")
	ax1.set_title("Smoothed Score per Episode")
	ax1.set_xlabel("Episode")
	ax1.set_ylabel("Score")
	ax1.grid(True)

	ax2.plot(episodes[len(episodes) - len(rolling_losses):], rolling_losses, label=f"Avg Loss ({window}-episode window)", color="orange")
	ax2.set_title("Smoothed Loss per Episode")
	ax2.set_xlabel("Episode")
	ax2.set_ylabel("Loss")
	ax2.grid(True)

	plt.tight_layout()
	plt.savefig("training_metrics.png")

	elapsed = time.time() - start_time
	minutes, seconds = divmod(int(elapsed), 60)
	print(f"\nTotal training time: {minutes} minutes and {seconds} seconds")

	print(f"\nBest score from a single episode in training: {best_score}")
		
if __name__ == "__main__":
	try:
		model_path = sys.argv[1] if len(sys.argv) > 1 else None # assign model path if given or set to None if not
		train(model_path)
	except KeyboardInterrupt:
		print("\nTraining interrupted by user.")
		pygame.quit()
