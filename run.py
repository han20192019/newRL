# a case for implementing OPE of the IPWLearner using synthetic bandit data
# import open bandit pipeline (obp)
from obp.dataset import SyntheticBanditDataset
from obp.policy.offline import NNPolicyLearner

# (1) Generate Synthetic Bandit Data
dataset = SyntheticBanditDataset(n_actions=10, reward_type="binary")
bandit_feedback_train = dataset.obtain_batch_bandit_feedback(n_rounds=1000)
bandit_feedback_test = dataset.obtain_batch_bandit_feedback(n_rounds=1000)

# (2) Off-Policy Learning
eval_policy = NNPolicyLearner(n_actions=dataset.n_actions, dim_context=dataset.dim_context)
print("????")
print(dataset.dim_context)
eval_policy.fit(
    context=bandit_feedback_train["context"],
    action=bandit_feedback_train["action"],
    reward=bandit_feedback_train["reward"]
)
#let N be the total number of generated pairs, context is N*dim_context, reward is N*1
#action_dist = eval_policy.predict(context=bandit_feedback_test["context"])
