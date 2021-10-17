"""Offline Bandit Algorithms."""
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from scipy.special import softmax
import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
import torch.optim as optim




@dataclass
class NNPolicyLearner():
    dim_context = 0
    n_actions = 0
    represent_model = None
    nn_model = None
    policy_model = None

    learning_rate_init: float = 0.0001
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-8
    alpha: float = 0.0001
    hidden_layer_size: Tuple[int, ...] = (3,)
    max_iter: int = 5


    def __init__(self, n_actions, dim_context) -> None:
        self.dim_context = dim_context
        self.n_actions = n_actions

        activation_layer = nn.ReLU
        layer_list = []
        layer_list_policy = []
        input_size = self.dim_context

        self.represent_model = nn.Linear(self.dim_context, self.dim_context)
        layer_list.append(("represent", self.represent_model))
        
        for i, h in enumerate(self.hidden_layer_size):
            layer_list.append(("l{}".format(i), nn.Linear(input_size, h)))
            layer_list.append(("a{}".format(i), activation_layer()))
            input_size = h
        layer_list.append(("output", nn.Linear(input_size, self.n_actions)))
        self.nn_model = nn.Sequential(OrderedDict(layer_list))
         #self.nn_model is f, produce a vector with dim n_actions representing the rewards for each of the actions

        #layer_list_policy.append(("representation", self.represent_model))  
        print(input_size)
        print("!")  
        layer_list_policy.append(("linear", nn.Linear(self.dim_context, self.n_actions)))
        layer_list_policy.append(("policy",nn.Softmax(dim = 0)))
        self.policy_model = nn.Sequential(OrderedDict(layer_list_policy))
        #self.policy_model produces a vector of dim n_actions representing the p for each action, sum of them should be 1

    def fit(
        self,
        context: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
    ) -> None:
        
        optimizer1 = optim.Adam(
            self.nn_model.parameters(),
            lr=self.learning_rate_init,
            betas=(self.beta_1, self.beta_2),
            eps=self.epsilon,
            weight_decay=self.alpha,
        )
        optimizer2 = optim.Adam(
            self.policy_model.parameters(),
            lr=self.learning_rate_init,
            betas=(self.beta_1, self.beta_2),
            eps=self.epsilon,
            weight_decay=self.alpha,
        )
        
        context = torch.from_numpy(context).float()
        action = torch.from_numpy(action).float()
        reward = torch.from_numpy(reward).float()

        for _ in range(self.max_iter):
            self.nn_model.train()
            print("reach")
            optimizer1.zero_grad()
            lossnew =  self.calculate_loss_for_nn(context, action, reward)
            loss = -lossnew 
            loss.backward()
            optimizer1.step()
            
            self.policy_model.train()
            optimizer2.zero_grad()
            lossnew =  self.calculate_loss_for_policy(context, action, reward)
            loss = lossnew 
            loss.backward()
            optimizer2.step()


    def calculate_loss_for_nn(
        self,
        context: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
    ):
        #context = torch.from_numpy(context)
        #what is P(x)?
        #haven't include the representation
        print("!!!")
        numsample = list(context.size())[0]
        dim_action = self.n_actions
        loss1 = 0
        for i in range(0, numsample):
            a = np.zeros(dim_action)
            #print(action[i].numpy())
            a[int(action[i].numpy())] = 1
            a = torch.from_numpy(a)
            with torch.no_grad():
                b = self.policy_model(context[i])
            loss1 += torch.linalg.norm(a-b)

        loss2 = 0
        for i in range(0, numsample):
            with torch.no_grad():
                r1 = self.policy_model(context[i])
            r2 = self.nn_model(context[i])
            temp = torch.dot(r1, r2)
            loss2 += temp
        
        loss3 = 0
        for i in range(0, numsample):
            predict_r = (self.nn_model(context[i]))[int(action[i].numpy())]
            loss3 += (predict_r - reward[i])*(predict_r - reward[i])
        loss3 /= numsample
        return loss1 + loss2 + loss3

    def calculate_loss_for_policy(
            self,
            context: torch.Tensor,
            action: torch.Tensor,
            reward: torch.Tensor,
        ):
        #context = torch.from_numpy(context)
        #what is P(x)?
        #haven't include the representation
        print("!!!")
        numsample = list(context.size())[0]
        dim_action = self.n_actions
        loss1 = 0
        for i in range(0, numsample):
            a = np.zeros(dim_action)
            #print(action[i].numpy())
            a[int(action[i].numpy())] = 1
            a = torch.from_numpy(a)
            b = self.policy_model(context[i])
            loss1 += torch.linalg.norm(a-b)

        loss2 = 0
        for i in range(0, numsample):
            r1 = self.policy_model(context[i])
            with torch.no_grad():
                r2 = self.nn_model(context[i])
            temp = torch.dot(r1, r2)
            loss2 += temp
        
        loss3 = 0
        for i in range(0, numsample):
            with torch.no_grad():
                predict_r = (self.nn_model(context[i]))[int(action[i].numpy())]
            loss3 += (predict_r - reward[i])*(predict_r - reward[i])
        loss3 /= numsample
        return loss1 + loss2 + loss3
