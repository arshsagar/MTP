import numpy as np
import seaborn as sns
from multiagent.core import World, Agent, Landmark, Wall
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def __init__(self, mode='train'):
        self.mode = mode

    def make_world(self): # check scenario.py
        world = World()
        # set any world properties first
        world.dim_c = 2
        world.treasure_colors = np.array(sns.color_palette(n_colors=2))
        num_good_agents = 0
        num_uavs = 2
        num_targets = num_uavs
        num_packages = 2
        # add agents
        world.agents = [Agent() for i in range(num_uavs)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.uav = True
            agent.size = 0.03
            agent.accel = 3.0
            agent.max_speed = 1.0
            agent.target = None
            # agent.ghost = True
        #############################
        # add landmarks             #
        # check line 109 of core.py #
        #############################
        # world.walls = [Wall() for i in range(9)]
        # world.obstacles = [Landmark() for i in range(8)]
        # for i, ob in enumerate(world.obstacles):
        #     ob.name = 'obstacle %d' % i
        #     ob.collide = True
        #     ob.movable = False
        #     ob.size = 0.15
        #     ob.boundary = True

        if self.mode == 'train':
            print('>>>>>> TRAIN')
            world.targets = [Landmark() for i in range(num_targets)]
            for i, target in enumerate(world.targets):
                target.name = 'target %d' % i
                target.collide = True
                target.movable = False
                target.size = 0.02
                target.boundary = False
                target.status = 'idle'
                target.item = None
        else:
            print('>>>>>> TEST')
            world.targets = [Landmark() for i in range(num_targets)]
            for i, target in enumerate(world.targets):
                target.name = 'target %d' % i
                target.collide = True
                target.movable = False
                target.size = 0.00001
                target.boundary = False
                target.status = 'idle'
                target.item = None

            world.p_locations = [Landmark() for i in range(num_packages)]
            for i, p_location in enumerate(world.p_locations):
                p_location.name = 'package %d' % i
                p_location.collide = False
                p_location.movable = False
                p_location.size = 0.02
                p_location.boundary = False
                p_location.collect = True
            world.d_locations = [Landmark() for i in range(num_packages)]
            for i, d_location in enumerate(world.d_locations):
                d_location.name = 'package %d' % i
                d_location.collide = False
                d_location.movable = False
                d_location.size = 0.02
                d_location.boundary = False
                d_location.deposit = True


        self.reset_world(world)
        return world
    ############################################################################
    ############################################################################
    def reset_world(self, world):
        # self.set_walls(world)
        # self.set_obs(world)
        type_ = [0,1]
        if self.mode == 'train':
            # random properties for agents
            for i, agent in enumerate(world.agents):
                agent.color = np.array([0.70, 0.70, 0.70])
            # random properties for targets
            world.targets[0].color = np.array([0.25, 0.25, 0.25])
            world.targets[1].color = np.array([0.85, 0.45, 0.25])

            # set random initial states
            for i, agent in enumerate(world.agents):
                val = np.random.uniform(-0.9, +0.9, world.dim_p)
                while(self.is_nofly(world, val)):
                    val = np.random.uniform(-0.9, +0.9, world.dim_p)
                agent.state.p_pos = val
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)
                agent.target = [world.targets[i]]
            for i, landmark in enumerate(world.targets):
                if not landmark.boundary:
                    val = np.random.uniform(-0.9, +0.9, world.dim_p)
                    while(self.is_nofly(world, val)):
                        val = np.random.uniform(-0.9, +0.9, world.dim_p)
                    landmark.state.p_pos = val
                    landmark.state.p_vel = np.zeros(world.dim_p)

        else:
            for i, agent in enumerate(world.agents):
                agent.color = np.array([0.70, 0.70, 0.70])
            for i, target in enumerate(world.targets):
                target.status = 'idle'
                target.item = None
                target.color = np.array([255, 255, 255])
            for i, p_location in enumerate(world.p_locations):
                p_location.collect = True
                p_location.color = world.treasure_colors[type_[i]]
            for i, d_location in enumerate(world.d_locations):
                d_location.deposit = True
                d_location.color = world.treasure_colors[type_[i]]

            for i, p_location in enumerate(world.p_locations):
                val = np.random.uniform(-0.9, +0.9, world.dim_p)
                while(self.is_nofly(world, val)):
                    val = np.random.uniform(-0.9, +0.9, world.dim_p)
                p_location.state.p_pos = val
                p_location.state.p_vel = np.zeros(world.dim_p)
            for i, d_location in enumerate(world.d_locations):
                val = np.random.uniform(-0.9, +0.9, world.dim_p)
                while(self.is_nofly(world, val)):
                    val = np.random.uniform(-0.9, +0.9, world.dim_p)
                d_location.state.p_pos = val
                d_location.state.p_vel = np.zeros(world.dim_p)
            for i, agent in enumerate(world.agents):
                val = np.random.uniform(-0.9, +0.9, world.dim_p)
                while(self.is_nofly(world, val)):
                    val = np.random.uniform(-0.9, +0.9, world.dim_p)
                agent.state.p_pos = val
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)
                agent.target = [world.targets[i]]
            for i, target in enumerate(world.targets):
                collects = self.collects(world)
                if len(collects) > 0:
                    target.state.p_pos = collects[0].state.p_pos
                    collects[0].collect = False
                    target.status = 'pick'
                    target.item = collects[0].name
                else:
                    target.state.p_pos = [0,0]
                    target.status = 'idle'
                    target.item = None

    ############################################################################
    ############################################################################
    def post_step(self, world):
        if self.mode == 'train':
            for UAV in self.uavs(world):
                for t in UAV.target:
                    if self.is_collision(UAV, t):
                        val = np.random.uniform(-0.9, +0.9, world.dim_p)
                        while(self.is_nofly(world, val)):
                            val = np.random.uniform(-0.9, +0.9, world.dim_p)
                        t.state.p_pos = val
                        UAV.color = t.color
                        print('                                             ITEM PICKED')
                        break
        else:
            for UAV in self.uavs(world):
                for target in UAV.target:
                    if target.status == 'pick':
                        for collect in world.p_locations:
                            if target.item == collect.name and self.is_collision(UAV, collect):
                                collect.state.p_pos = [-999,-999]
                                target.status = 'drop'

                                for deposit in world.d_locations:
                                    if target.item == deposit.name:
                                        target.state.p_pos = deposit.state.p_pos
                                        deposit.deposit = False
                                break
                    elif target.status == 'drop':
                        for deposit in world.d_locations:
                            if target.item == deposit.name and self.is_collision(UAV, deposit):
                                deposit.state.p_pos = [-999,-999]

                                collects = self.collects(world)
                                if len(collects) > 0:
                                    target.state.p_pos = collects[0].state.p_pos
                                    collects[0].collect = False
                                    target.status = 'pick'
                                    target.item = collects[0].name
                                else:
                                    target.state.p_pos = [0,0]
                                    target.status = 'idle'
                                    target.item = None
                                break
                    else:
                        target.state.p_pos = [0,0]
                        target.status = 'idle'
                        target.item = None

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            print('yes')
            collisions = 0
            # for a in self.landmarks(world):
            #     if self.is_collision(a, agent):
            #         collisions += 1
            return collisions
        else:
            print('benchmark_data')
            return 0


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def uavs(self, world):
        return [agent for agent in world.agents if agent.uav]

    def collects(self, world):
        return [a for a in world.p_locations if a.collect]

    def deposits(self, world):
        return [a for a in world.d_locations if a.deposit]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward, agentCollisions, wallCollisions = self.uav_reward(agent, world)
        return main_reward, agentCollisions, wallCollisions

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by uavs
        # rew = 0
        # shape = True
        # uavs = self.uavs(world)
        # if shape:  # reward can optionally be shaped (increased reward for increased distance from uav)
        #     for adv in uavs:
        #         rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
        # if agent.collide:
        #     for a in uavs:
        #         if self.is_collision(a, agent):
        #             rew -= 10
        #
        # # agents are penalized for exiting the screen, so that they can be caught by the uavs
        # def bound(x):
        #     if x < 0.9:
        #         return 0
        #     if x < 1.0:
        #         return (x - 0.9) * 10
        #     return min(np.exp(2 * x - 2), 10)
        # for p in range(world.dim_p):
        #     x = abs(agent.state.p_pos[p])
        #     rew -= bound(x)

        return rew

    def uav_reward(self, agent, world):
        # uavs are rewarded for collisions with agents
        C = 1
        rew = 0
        shape = True

        target = agent.target[0]
        rew -= (C/10) * np.sqrt(np.sum(np.square(agent.state.p_pos - target.state.p_pos)))
        if agent.collide:
            if self.is_collision(agent, target):
                rew += (10*C)
        # agents = agent.target
        # uavs = self.uavs(world)
        # if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
        #     for adv in uavs:
        #         rew -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in agents])
        # if agent.collide:
        #     for ag in agents:
        #         for adv in uavs:
        #             if self.is_collision(ag, adv):
        #                 rew += 10
        agentCollisions = 0
        wallCollisions = 0
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= (1*C)
                    agentCollisions += 1
        # if agent.collide:
        #     for ob in world.obstacles:
        #         if self.is_collision(ob, agent):
        #             rew -= (1*C)
        #             wallCollisions += 1
        # if self.is_nofly_region(world, agent.state.p_pos, agent.size):
        #     rew -= (5*C)

        # if agent.collide:
        #     for i, wall in enumerate(world.walls):
        #         mag = world.get_wall_collision_force_mag(agent, wall)
        #         if mag is not None and mag != 0.0:
        #             rew -= (0.1*C)
        # mag_list = [world.get_wall_collision_force_mag(agent, wall) for wall in world.walls if world.get_wall_collision_force_mag(agent, wall) is not None]
        # if len(mag_list) > 0:
        #     magnitude = max(mag_list)
        #     rew -= abs(magnitude)

        return rew, agentCollisions, wallCollisions

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in agent.target:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if other.uav:
                other_vel.append(other.state.p_vel)

        # wall_endpts = []
        # for wall in world.walls:
        #     corners = ((wall.axis_pos - 0.5 * wall.width, wall.endpoints[0]),
        #                (wall.axis_pos - 0.5 * wall.width, wall.endpoints[1]),
        #                (wall.axis_pos + 0.5 * wall.width, wall.endpoints[1]),
        #                (wall.axis_pos + 0.5 * wall.width, wall.endpoints[0]))
        #     pos = []
        #     pos.append((corners[0][1]-0.05))
        #     pos.append((corners[1][1]+0.05))
        #     pos.append((corners[0][0]-0.05))
        #     pos.append((corners[2][0]+0.05))
        #     wall_endpts.append(pos)

        # obstacle_pos = []
        # for ob in world.obstacles:
        #     obstacle_pos.append(ob.state.p_pos - agent.state.p_pos)

        observation = np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)
        # observation = np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel + wall_endpts)
        # observation = np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel + obstacle_pos)
        # print(observation)
        return observation

    # def set_walls(self, world):
    #     # world.walls[0].endpoints = [-1, 1]
    #     # world.walls[0].axis_pos = 2
    #     # world.walls[0].width = 0.006
    #     # world.walls[1].endpoints = [-1, 1]
    #     # world.walls[1].axis_pos = 2
    #     # world.walls[1].width = 0.006
    #     # world.walls[1].orient = 'V'
    #     # world.walls[2].endpoints = [-1, 1]
    #     # world.walls[2].axis_pos = -2
    #     # world.walls[2].width = 0.006
    #     # world.walls[3].endpoints = [-1, 1]
    #     # world.walls[3].axis_pos = -2
    #     # world.walls[3].width = 0.006
    #     # world.walls[3].orient = 'V'
    #     world.walls[0].endpoints = [-0.15, 0.15]
    #     world.walls[0].axis_pos = 10.0
    #     world.walls[0].width = 0.2
    #     world.walls[0].color = np.array([255, 255, 0])
    #     world.walls[0].hard = False
    #     ###########################
    #     world.walls[1].endpoints, world.walls[1].axis_pos, world.walls[1].width = [-0.85, -0.55], 0.8, 0.2
    #     world.walls[2].endpoints, world.walls[2].axis_pos, world.walls[2].width = [-0.15, 0.15], 0.8, 0.2
    #     world.walls[3].endpoints, world.walls[3].axis_pos, world.walls[3].width = [0.55, 0.85], 0.8, 0.2
    #
    #     world.walls[4].endpoints, world.walls[4].axis_pos, world.walls[4].width = [-0.85, -0.55], 0.0, 0.2
    #     world.walls[5].endpoints, world.walls[5].axis_pos, world.walls[5].width = [0.55, 0.85], 0.0, 0.2
    #
    #     world.walls[6].endpoints, world.walls[6].axis_pos, world.walls[6].width = [-0.85, -0.55], -0.8, 0.2
    #     world.walls[7].endpoints, world.walls[7].axis_pos, world.walls[7].width = [-0.15, 0.15], -0.8, 0.2
    #     world.walls[8].endpoints, world.walls[8].axis_pos, world.walls[8].width = [0.55, 0.85], -0.8, 0.2
    #
    #     # world.walls[1].endpoints, world.walls[1].axis_pos, world.walls[1].width = [-0.85, -0.55], np.random.uniform(-1, +1), 0.2
    #     # world.walls[2].endpoints, world.walls[2].axis_pos, world.walls[2].width = [-0.15, 0.15], np.random.uniform(-1, +1), 0.2
    #     # world.walls[3].endpoints, world.walls[3].axis_pos, world.walls[3].width = [0.55, 0.85], np.random.uniform(-1, +1), 0.2
    #     #
    #     # world.walls[4].endpoints, world.walls[4].axis_pos, world.walls[4].width = [-0.85, -0.55], np.random.uniform(-1, +1), 0.2
    #     # world.walls[5].endpoints, world.walls[5].axis_pos, world.walls[5].width = [0.55, 0.85], np.random.uniform(-1, +1), 0.2
    #     #
    #     # world.walls[6].endpoints, world.walls[6].axis_pos, world.walls[6].width = [-0.85, -0.55], np.random.uniform(-1, +1), 0.2
    #     # world.walls[7].endpoints, world.walls[7].axis_pos, world.walls[7].width = [-0.15, 0.15], np.random.uniform(-1, +1), 0.2
    #     # world.walls[8].endpoints, world.walls[8].axis_pos, world.walls[8].width = [0.55, 0.85], np.random.uniform(-1, +1), 0.2

    def set_obs(self, world):
        # world.obstacles[0].state.p_pos = [-0.85, +0.85]
        # world.obstacles[1].state.p_pos = [0, +0.85]
        # world.obstacles[2].state.p_pos = [0.85, +0.85]
        # world.obstacles[3].state.p_pos = [-0.85, 0]
        # world.obstacles[4].state.p_pos = [0.85, 0]
        # world.obstacles[5].state.p_pos = [-0.85, -0.85]
        # world.obstacles[6].state.p_pos = [0, -0.85]
        # world.obstacles[7].state.p_pos = [+0.85, -0.85]
        for i, ob in enumerate(world.obstacles):
            val = np.random.uniform(-1, +1, world.dim_p)
            ob.state.p_pos = val
            ob.state.p_vel = np.zeros(world.dim_p)
            ob.color = np.array([0.85, 0, 0])

    def is_nofly(self, world, val):
        # for i, wall in enumerate(world.walls):
        #     if wall.orient == 'H':
        #         corners = ((wall.axis_pos - 0.5 * wall.width, wall.endpoints[0]),
        #                    (wall.axis_pos - 0.5 * wall.width, wall.endpoints[1]),
        #                    (wall.axis_pos + 0.5 * wall.width, wall.endpoints[1]),
        #                    (wall.axis_pos + 0.5 * wall.width, wall.endpoints[0]))
        #         if val[0] >= (corners[0][1]-0.1) and val[0] <= (corners[1][1]+0.1) and val[1] >= (corners[0][0]-0.1) and val[1] <= (corners[2][0]+0.1):
        #             return True
        # return False

        # for i, ob in enumerate(world.obstacles):
        #     delta_pos = ob.state.p_pos - val
        #     dist = np.sqrt(np.sum(np.square(delta_pos)))
        #     dist_min = ob.size + 0.03
        #     if dist < dist_min:
        #         return True
        return False

    # def is_nofly_region(self, world, val, size):
    #     for i, wall in enumerate(world.walls):
    #         if wall.orient == 'H':
    #             corners = ((wall.axis_pos - 0.5 * wall.width, wall.endpoints[0]),
    #                        (wall.axis_pos - 0.5 * wall.width, wall.endpoints[1]),
    #                        (wall.axis_pos + 0.5 * wall.width, wall.endpoints[1]),
    #                        (wall.axis_pos + 0.5 * wall.width, wall.endpoints[0]))
    #             if val[0] >= (corners[0][1]-0.05) and val[0] <= (corners[1][1]+0.05) and val[1] >= (corners[0][0]-0.05) and val[1] <= (corners[2][0]+0.05):
    #                 return True
    #     return False
