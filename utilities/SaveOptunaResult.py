import optuna
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_parallel_coordinate
import plotly

SEED = 420
VERSION = "v3_3_0"
# VERSION = "v3_3_0_2"
# VERSION = "v3_3_1_0_old"
# VERSION = "v3_3_1_0"
# VERSION = "v3_5_0_0"
# VERSION = "v3_5_0_1"
# VERSION = "v3_5_1_0"


study = optuna.create_study(
    storage="sqlite:///runs/optuna_study_database/TransformerHyperparas.db",  # Specify the storage URL here.
    sampler=optuna.samplers.TPESampler(seed=SEED),
    pruner=optuna.pruners.HyperbandPruner(
        min_resource=1, reduction_factor=3
    ),
    study_name="{}".format(VERSION),
    direction="minimize",
    load_if_exists=True,
)

fig = plot_intermediate_values(study)
fig.update_layout(showlegend=True)
fig.update_xaxes(title_font=dict(size=25))
fig.update_yaxes(title="Validation Loss", title_font=dict(size=25))
fig.update_layout(font=dict(family='Times New Roman'))


# fig = plot_parallel_coordinate(study, params=["lr_init", "n_units_l0"])
# fig = plot_parallel_coordinate(study, target_name="Validation<br>Loss")
# fig.update_layout(font=dict(family='Times New Roman', size=18))

# custom_colorscale = plotly.colors.sequential.Bluered_r
# # custom_colorscale = plotly.colors.sequential.Plotly3_r
# for trace in fig.data:
#     trace.line.colorscale = custom_colorscale


fig.layout.title = None
fig.show()
# fig.write_image(f"Optuna_figs/parallel_coordinate_{VERSION}.pdf", format="pdf")
# fig.write_image(f"Optuna_figs/parallel_coordinate_{VERSION}.pdf", format="pdf", width=600, height=400)
# fig.write_image(f"Optuna_figs/parallel_coordinate_{VERSION}.pdf", format="pdf", width=480, height=360)
# fig.write_image(f"Optuna_figs/intermediate_values_{VERSION}.pdf", format="pdf")
# fig.write_image(f"Optuna_figs/intermediate_values_{VERSION}.pdf", format="pdf", width=750, height=750)