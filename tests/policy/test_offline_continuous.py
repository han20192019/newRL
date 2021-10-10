import numpy as np
import pytest
import torch

from obp.policy.offline_continuous import ContinuousNNPolicyLearner


# dim_context, pg_method, bandwidth, output_space, hidden_layer_size, activation, solver, alpha,
# batch_size, learning_rate_init, max_iter, shuffle, random_state, tol, momentum, nesterovs_momentum,
# early_stopping, validation_fraction, beta_1, beta_2, epsilon, n_iter_no_change, q_func_estimator_hyperparams, description
invalid_input_of_nn_policy_learner_init = [
    (
        0,  #
        "ipw",
        0.1,
        (-10, 10),
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        "auto",
        0.0001,
        200,
        True,
        123,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        None,
        "`dim_context`= 0, must be >= 1",
    ),
    (
        10,
        "None",  #
        2,
        (-10, 10),
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        "auto",
        0.0001,
        200,
        True,
        123,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        None,
        "pg_method must be one of 'dgp', 'ipw', or 'dr'",
    ),
    (
        10,
        "ipw",
        -0.1,  #
        (-10, 10),
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        "auto",
        0.0001,
        200,
        True,
        123,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        None,
        "`bandwidth`= -0.1, must be > 0",
    ),
    (
        10,
        "ipw",
        0.1,
        ("", ""),  #
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        "auto",
        0.0001,
        200,
        True,
        123,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        None,
        "output_space must be tuple of integers or floats",
    ),
    (
        10,
        "ipw",
        0.1,
        (-10, 10),
        (100, ""),  #
        "relu",
        "adam",
        0.001,
        "auto",
        0.0001,
        200,
        True,
        123,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        None,
        "hidden_layer_size must be tuple of positive integers",
    ),
    (
        10,
        "ipw",
        0.1,
        (-10, 10),
        (100, 50, 100),
        "None",  #
        "adam",
        0.001,
        "auto",
        0.0001,
        200,
        True,
        123,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        None,
        "activation must be one of 'identity', 'logistic', 'tanh', 'relu', or 'elu'",
    ),
    (
        10,
        "ipw",
        0.1,
        (-10, 10),
        (100, 50, 100),
        "relu",
        "None",  #
        0.001,
        "auto",
        0.0001,
        200,
        True,
        123,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        None,
        "solver must be one of 'adam', 'adagrad', or 'sgd'",
    ),
    (
        10,
        "ipw",
        0.1,
        (-10, 10),
        (100, 50, 100),
        "relu",
        "adam",
        -1.0,  #
        "auto",
        0.0001,
        200,
        True,
        123,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        None,
        "`alpha`= -1.0, must be >= 0.0",
    ),
    (
        10,
        "ipw",
        0.1,
        (-10, 10),
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        0,  #
        0.0001,
        200,
        True,
        123,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        None,
        "batch_size must be a positive integer or 'auto'",
    ),
    (
        10,
        "ipw",
        0.1,
        (-10, 10),
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        "auto",
        0.0,  #
        200,
        True,
        123,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        None,
        "`learning_rate_init`= 0.0, must be > 0.0",
    ),
    (
        10,
        "ipw",
        0.1,
        (-10, 10),
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        "auto",
        0.0001,
        0,  #
        True,
        123,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        None,
        "`max_iter`= 0, must be >= 1",
    ),
    (
        10,
        "ipw",
        0.1,
        (-10, 10),
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        "auto",
        0.0001,
        200,
        None,  #
        123,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        None,
        "shuffle must be a bool",
    ),
    (
        10,
        "ipw",
        0.1,
        (-10, 10),
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        "auto",
        0.0001,
        200,
        True,
        "",  #
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        None,
        "'' cannot be used to seed",
    ),
    (
        10,
        "ipw",
        0.1,
        (-10, 10),
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        "auto",
        0.0001,
        200,
        True,
        123,
        -1.0,  #
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        None,
        "`tol`= -1.0, must be > 0.0",
    ),
    (
        10,
        "ipw",
        0.1,
        (-10, 10),
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        "auto",
        0.0001,
        200,
        True,
        123,
        1e-4,
        2.0,  #
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        None,
        "`momentum`= 2.0, must be <= 1.0",
    ),
    (
        10,
        "ipw",
        0.1,
        (-10, 10),
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        "auto",
        0.0001,
        200,
        True,
        123,
        1e-4,
        0.9,
        "",  #
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        None,
        "nesterovs_momentum must be a bool",
    ),
    (
        10,
        "ipw",
        0.1,
        (-10, 10),
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        "auto",
        0.0001,
        200,
        True,
        123,
        1e-4,
        0.9,
        True,
        None,  #
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        None,
        "early_stopping must be a bool",
    ),
    (
        10,
        "ipw",
        0.1,
        (-10, 10),
        (100, 50, 100),
        "relu",
        "lbfgs",  #
        0.001,
        "auto",
        0.0001,
        200,
        True,
        123,
        1e-4,
        0.9,
        True,
        True,  #
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        None,
        "solver must be one of 'adam', 'adagrad', or 'sgd',",
    ),
    (
        10,
        "ipw",
        0.1,
        (-10, 10),
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        "auto",
        0.0001,
        200,
        True,
        123,
        1e-4,
        0.9,
        True,
        True,
        2.0,  #
        0.9,
        0.999,
        1e-8,
        10,
        None,
        "`validation_fraction`= 2.0, must be <= 1.0",
    ),
    (
        10,
        "ipw",
        0.1,
        (-10, 10),
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        "auto",
        0.0001,
        200,
        True,
        123,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        2.0,  #
        0.999,
        1e-8,
        10,
        None,
        "`beta_1`= 2.0, must be <= 1.0",
    ),
    (
        10,
        "ipw",
        0.1,
        (-10, 10),
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        "auto",
        0.0001,
        200,
        True,
        123,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        2.0,  #
        1e-8,
        10,
        None,
        "`beta_2`= 2.0, must be <= 1.0",
    ),
    (
        10,
        "ipw",
        0.1,
        (-10, 10),
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        "auto",
        0.0001,
        200,
        True,
        123,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        -1.0,  #
        10,
        None,
        "`epsilon`= -1.0, must be >= 0.0",
    ),
    (
        10,
        "ipw",
        0.1,
        (-10, 10),
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        "auto",
        0.0001,
        200,
        True,
        123,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        0,  #
        None,
        "`n_iter_no_change`= 0, must be >= 1",
    ),
    (
        10,
        "ipw",
        0.1,
        (-10, 10),
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        "auto",
        0.0001,
        200,
        True,
        123,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        "",  #
        "q_func_estimator_hyperparams must be a dict,",
    ),
]

valid_input_of_nn_policy_learner_init = [
    (
        10,
        "ipw",
        0.1,
        (-10, 10),
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        "auto",
        0.0001,
        200,
        True,
        123,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        None,
        "valid input",
    ),
    (
        10,
        "dpg",
        None,
        (-10, 10),
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        "auto",
        0.0001,
        200,
        True,
        123,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        None,
        "valid input",
    ),
    (
        10,
        "ipw",
        0.1,
        (-10, 10),
        (100, 50, 100),
        "logistic",
        "sgd",
        0.001,
        50,
        0.0001,
        200,
        True,
        None,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        {},
        "valid input",
    ),
]


@pytest.mark.parametrize(
    "dim_context, pg_method, bandwidth, output_space, hidden_layer_size, activation, solver, alpha, batch_size, learning_rate_init, max_iter, shuffle, random_state, tol, momentum, nesterovs_momentum, early_stopping, validation_fraction, beta_1, beta_2, epsilon, n_iter_no_change, q_func_estimator_hyperparams, description",
    invalid_input_of_nn_policy_learner_init,
)
def test_nn_policy_learner_init_using_invalid_inputs(
    dim_context,
    pg_method,
    bandwidth,
    output_space,
    hidden_layer_size,
    activation,
    solver,
    alpha,
    batch_size,
    learning_rate_init,
    max_iter,
    shuffle,
    random_state,
    tol,
    momentum,
    nesterovs_momentum,
    early_stopping,
    validation_fraction,
    beta_1,
    beta_2,
    epsilon,
    n_iter_no_change,
    q_func_estimator_hyperparams,
    description,
):
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = ContinuousNNPolicyLearner(
            dim_context=dim_context,
            pg_method=pg_method,
            bandwidth=bandwidth,
            output_space=output_space,
            hidden_layer_size=hidden_layer_size,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            shuffle=shuffle,
            random_state=random_state,
            tol=tol,
            momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            n_iter_no_change=n_iter_no_change,
            q_func_estimator_hyperparams=q_func_estimator_hyperparams,
        )


@pytest.mark.parametrize(
    "dim_context, pg_method, bandwidth, output_space, hidden_layer_size, activation, solver, alpha, batch_size, learning_rate_init, max_iter, shuffle, random_state, tol, momentum, nesterovs_momentum, early_stopping, validation_fraction, beta_1, beta_2, epsilon, n_iter_no_change, q_func_estimator_hyperparams, description",
    valid_input_of_nn_policy_learner_init,
)
def test_nn_policy_learner_init_using_valid_inputs(
    dim_context,
    pg_method,
    bandwidth,
    output_space,
    hidden_layer_size,
    activation,
    solver,
    alpha,
    batch_size,
    learning_rate_init,
    max_iter,
    shuffle,
    random_state,
    tol,
    momentum,
    nesterovs_momentum,
    early_stopping,
    validation_fraction,
    beta_1,
    beta_2,
    epsilon,
    n_iter_no_change,
    q_func_estimator_hyperparams,
    description,
):
    nn_policy_learner = ContinuousNNPolicyLearner(
        dim_context=dim_context,
        pg_method=pg_method,
        bandwidth=bandwidth,
        output_space=output_space,
        hidden_layer_size=hidden_layer_size,
        activation=activation,
        solver=solver,
        alpha=alpha,
        batch_size=batch_size,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        shuffle=shuffle,
        random_state=random_state,
        tol=tol,
        momentum=momentum,
        nesterovs_momentum=nesterovs_momentum,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
        n_iter_no_change=n_iter_no_change,
        q_func_estimator_hyperparams=q_func_estimator_hyperparams,
    )
    assert isinstance(nn_policy_learner, ContinuousNNPolicyLearner)


def test_nn_policy_learner_create_train_data_for_opl():
    context = np.ones((100, 2), dtype=np.int32)
    action = np.zeros(100, dtype=np.int32)
    reward = np.ones((100,), dtype=np.float32)
    pscore = np.array([0.5] * 100, dtype=np.float32)

    learner1 = ContinuousNNPolicyLearner(dim_context=2, pg_method="dpg")
    training_loader, validation_loader = learner1._create_train_data_for_opl(
        context=context,
        action=action,
        reward=reward,
        pscore=pscore,
    )

    assert isinstance(training_loader, torch.utils.data.DataLoader)
    assert validation_loader is None

    learner2 = ContinuousNNPolicyLearner(
        dim_context=2,
        pg_method="dpg",
        early_stopping=True,
    )

    training_loader, validation_loader = learner2._create_train_data_for_opl(
        context=context,
        action=action,
        reward=reward,
        pscore=pscore,
    )

    assert isinstance(training_loader, torch.utils.data.DataLoader)
    assert isinstance(validation_loader, torch.utils.data.DataLoader)


# context, action, reward, pscore, description
invalid_input_of_nn_policy_learner_fit = [
    (
        5,  #
        np.ones(5),
        np.ones(5),
        np.ones(5) * 0.5,
        "context must be 2D array",
    ),
    (
        np.ones(5),  #
        np.ones(5),
        np.ones(5),
        np.ones(5) * 0.5,
        "context must be 2D array",
    ),
    (
        np.ones((5, 2)),
        5,  #
        np.ones(5),
        np.ones(5) * 0.5,
        "action_by_behavior_policy must be 1D array",
    ),
    (
        np.ones((5, 2)),
        np.ones((5, 2)),  #
        np.ones(5),
        np.ones(5) * 0.5,
        "action_by_behavior_policy must be 1D array",
    ),
    (
        np.ones((5, 2)),
        np.ones(5),
        5,  #
        np.ones(5) * 0.5,
        "reward must be 1D array",
    ),
    (
        np.ones((5, 2)),
        np.ones(5),
        np.ones((5, 2)),  #
        np.ones(5) * 0.5,
        "reward must be 1D array",
    ),
    (
        np.ones((5, 2)),
        np.ones(5),
        np.ones(5),
        0.5,  #
        "pscore must be 1D array",
    ),
    (
        np.ones((5, 2)),
        np.ones(5),
        np.ones(5),
        np.ones((5, 2)) * 0.5,  #
        "pscore must be 1D array",
    ),
    (
        np.ones((4, 2)),  #
        np.ones(5),
        np.ones(5),
        np.ones(5) * 0.5,
        "Expected `context.shape[0]",
    ),
    (
        np.ones((5, 2)),
        np.ones(4),  #
        np.ones(5),
        np.ones(5) * 0.5,
        "Expected `context.shape[0]",
    ),
    (
        np.ones((5, 2)),
        np.ones(5),
        np.ones(4),  #
        np.ones(5) * 0.5,
        "Expected `context.shape[0]",
    ),
    (
        np.ones((5, 2)),
        np.ones(5),
        np.ones(5),
        np.arange(5) * 0.1,  #
        "pscore must be positive",
    ),
    (
        np.ones((5, 3)),  #
        np.ones(5),
        np.ones(5),
        np.ones(5) * 0.5,
        "Expected `context.shape[1]",
    ),
]

valid_input_of_nn_policy_learner_fit = [
    (
        np.ones((5, 2)),
        np.ones(5),
        np.ones(5),
        np.ones(5) * 0.5,
        "valid input (pscore is given)",
    ),
    (
        np.ones((5, 2)),
        np.ones(5),
        np.ones(5),
        None,
        "valid input (pscore is not given)",
    ),
]


@pytest.mark.parametrize(
    "context, action, reward, pscore, description",
    invalid_input_of_nn_policy_learner_fit,
)
def test_nn_policy_learner_fit_using_invalid_inputs(
    context,
    action,
    reward,
    pscore,
    description,
):
    with pytest.raises(ValueError, match=f"{description}*"):
        # set parameters
        dim_context = 2
        pg_method = "dpg"
        learner = ContinuousNNPolicyLearner(
            dim_context=dim_context, pg_method=pg_method
        )
        learner.fit(
            context=context,
            action=action,
            reward=reward,
            pscore=pscore,
        )


@pytest.mark.parametrize(
    "context, action, reward, pscore, description",
    valid_input_of_nn_policy_learner_fit,
)
def test_nn_policy_learner_fit_using_valid_inputs(
    context,
    action,
    reward,
    pscore,
    description,
):
    # set parameters
    dim_context = 2
    pg_method = "dpg"
    learner = ContinuousNNPolicyLearner(dim_context=dim_context, pg_method=pg_method)
    learner.fit(
        context=context,
        action=action,
        reward=reward,
        pscore=pscore,
    )


def test_nn_policy_learner_predict():
    # synthetic data
    context = np.ones((5, 2))
    action = np.ones(5)
    reward = np.ones(5)

    # set parameters
    dim_context = 2
    pg_method = "dpg"
    output_space = (-10, 10)
    learner = ContinuousNNPolicyLearner(
        dim_context=dim_context, pg_method=pg_method, output_space=output_space
    )
    learner.fit(
        context=context,
        action=action,
        reward=reward,
    )

    # shape error
    with pytest.raises(ValueError, match="context must be 2D array"):
        learner.predict(context=np.ones(5))

    with pytest.raises(ValueError, match="context must be 2D array"):
        learner.predict(context="np.ones(5)")

    # inconsistency between dim_context and context
    with pytest.raises(ValueError, match="Expected `context.shape[1]*"):
        learner.predict(context=np.ones((5, 3)))

    # check output shape
    predicted_actions = learner.predict(context=context)
    assert predicted_actions.shape[0] == context.shape[0]
    assert predicted_actions.ndim == 1
    assert np.all(output_space[0] <= predicted_actions) or np.all(
        predicted_actions <= output_space[1]
    )
