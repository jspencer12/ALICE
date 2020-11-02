from matplotlib import animation
import matplotlib.pyplot as plt
import gym 
import pyglet
import time
import numpy as np
"""
Ensure you have imagemagick installed with 
sudo apt-get install imagemagick
Open file in CLI with:
xgd-open <filelname>
"""
#class RenderActionWrapper(gym.Wrappers):
    
def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):
    #Mess with this to change frame size
    fig = plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    plt.axis('off')
    fig.tight_layout()
    patch = plt.imshow(frames[0])
    plt.show()
    #animate = lambda i: patch.set_data(frames[i])
    #gif = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    #gif.save(path + filename, writer='imagemagick', fps=20)

#Make gym env
env = gym.make('Acrobot-v1')

#Run the env
observation = env.reset()
frames = []
score_label = pyglet.text.Label('0000', font_size=36,
                x=20, y=480, anchor_x='left', anchor_y='top',
                color=(255,63,63,255))
for t in range(10):
    action = env.action_space.sample()
    
    #Render to frames buffer
    time.sleep(.1)
    
    env.render(mode="rgb_array")
    score_label.text = "Action: {: d}".format(action-1)
    score_label.draw()
    #pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
    #pyglet.gl.glClearColor((36+1.0)/256, (72+1.0)/256, (132+1.0)/256,1) #244884
    env.viewer.window.flip()
    arr = np.fromstring(pyglet.image.get_buffer_manager().get_color_buffer().get_image_data().get_data(), dtype=np.uint8, sep='')
    arr = arr.reshape(env.viewer.height, env.viewer.width, 4)[::-1, :, 0:3]
    print(arr.shape)
    frames.append(arr)
    #for i in range(arr.shape[0]):
     #   for j in range(arr.shape[1]):
      #      if (arr[i,j,:] == np.array([255,255,255])).all():
       #         arr[i,j,:] = np.array([36,72,132])
    _, _, done, _ = env.step(action)
    if done:
        break
env.close()
save_frames_as_gif(frames[1:])
