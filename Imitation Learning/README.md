# Imitation Learning
Imitation learning is an algorithm where you learn to *imitate* an expert. From the expert demonstrations of how any task is done, an imitation algorithm learns to predict what that expert would do, given any situation.

This is done by collecting trajectory---a tuple of (**state**, **action**) pair---from expert's demonstration and then training a neural network to predict what the expert did given the input data. In the trajectory, the *state* is the input to the network---what setting the expert is in---and the *action* is what the expert did in that setting. For e.g., while imitating to play mario, if the expert jumped when the turtle came close to it (as given by state), our imitation learning algorithm's objective is to learn to jump when the agent is put in that state.

## Behavior Cloning
Behavior cloning is a simple imitation learning technique where you learn to imitate expert by training a neural network on trajectories collected from expert demonstrations. An input to the network is the state and the output is the action from the trajectory.

A simple way to see behavioral cloning working is to train a simple Mountain Car to climb the mountain on itself.

### Simple Behavioral Cloning in Mountain Car Environment
A mountain car is a [Gymnasium](https://gymnasium.farama.org/environments/classic_control/mountain_car/) environment, where the robot's task is to give the right controls to be able to climb the mountain.

![Mountain Car Environment](mountain_car.gif)

| | type | |
|:-:|:-------:|:-:|
|State|Array([pos, vel])|position of the car along the x-axis (-1.2 to 0.6) and velocity (-0.07 to 0.07)|
|Action|int|0: accelerate left, 1: do nothing, 1: accelerate right|

Given any state, we need to learn what the action the agent should do.

#### Collecting expert's trajectory
To collect expert trajectory, we can either run the mountain car ourselves, or create a simple expert and then record the trajectory.

For this example, we create a simple expert which works in the following way: if the velocity of the car is negative, we push left, and if the velocity is positive, we push right to climb the hill. If the velocity is 0, we do nothing.
```python
PUSH_LEFT = 0
PUSH_RIGHT = 2
PUSH_NOOP = 1

def expert_policy(state):
    _, velocity = state
    if velocity > 0:
        return PUSH_RIGHT
    elif velocity < 0:
        return PUSH_LEFT
    else:
        return PUSH_NOOP
```
and collect the data. See the code to collect data [here](https://github.com/aabs7/Robot-Learning/blob/main/Imitation%20Learning/mountain_car/mc_expert_traj_gen.py).

In the code, we run the gym environment, and append the (state, action) tuple to a list of expert trajectory for 10000 different runs and then save it in pickle file.

#### Training a simple neural network to learn expert's behavior
The input to the network is the state (position, and velocity), and output is which action the car should take (i.e., 0, 1 or 2). Therefore, a simple network that can achieve this is:
```python
class BCModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
    def forward(self, x):
        return self.net(x)
```
We can train this model using cross entropy loss. See the code for training [here](https://github.com/aabs7/Robot-Learning/blob/main/Imitation%20Learning/mountain_car/mc_behavior_cloning_train.py) and testing [here](https://github.com/aabs7/Robot-Learning/blob/main/Imitation%20Learning/mountain_car/mc_behavior_cloning_test.py).

We can see that our simple behavior cloning algorithm works by learning from expert's demonstration.

[Add figure here]

This is a simple example because the state space consists of a tuple (position, velocity). It's very likeliy that the model has seen almost all of the state in the training dataset. This might not be true in the real-world.

### Behavioral Cloning in Car Racing Environment
Next, we move on to a little bit complicated problem where the state space is big, i.e., car racing environment.
![Car Racing Environment](car_racing.gif)
||type||
|:-:|:-:|:--:|
|State| Image (96x96x3)| Top down RGB image of car and race track
|Action|array([steer, gas, brake])|Continuous value actions. Steer (-1 to 1), gas (0 to 1), and brake (0, 1).

#### Collecting expert's trajectory
An easiest way to collect data for this is to drive ourselves. Therefore, we use python pygame library to get events from the keyboard and then construct actions. Below is a code to make actions from key pressed.

```python
def get_keyboard_action():
    steer = 0.0
    gas = 0.0
    brake = 0.0

    keys = pygame.key.get_pressed()

    if keys[pygame.K_LEFT]:
        steer = -0.5
    elif keys[pygame.K_RIGHT]:
        steer = 0.5
    if keys[pygame.K_UP]:
        gas = 0.2
    if keys[pygame.K_DOWN]:
        brake = 0.1

    return np.array([steer, gas, brake], dtype=np.float32)
```

Now, we collect the data by driving the car couple of times till completion and use this data to train the model. See the data collection script [here](https://github.com/aabs7/Robot-Learning/blob/main/Imitation%20Learning/car_racing/cr_expert_traj_gen.py).

#### Training the neural network and evaluating
We can use CNN to train the network using MSE Loss. See the code [here](https://github.com/aabs7/Robot-Learning/blob/main/Imitation%20Learning/car_racing/cr_behavior_cloning_train.py). Using the trained model, we can evaluate our approach using the trained network. See the code [here](https://github.com/aabs7/Robot-Learning/blob/main/Imitation%20Learning/car_racing/cr_behavior_cloning_test.py).

[Add figure here]

You can see that it sort of works. But it isn't working really well. More training data can definitely help but there's a deep rooted challenge with behavioral cloning that this example highlights.


### Challenges in Behavioral Cloning
Error aggregation is one of the main challenge in behavioral cloning. When the input is passed to a model, it outputs an action. By executing that action, the robot/agent could reach a state that the model has never seen before and so on. This results in error aggregation. In figure above, we can see that once the car goes off track, the model predicts actions that takes the car further away from track quickly.
People get around this error by using algorithms like DAgger, or by using more powerful models.
