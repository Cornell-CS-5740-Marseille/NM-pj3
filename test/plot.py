import numpy as np
import csv
from collections import *
from pygg import *
from wuutils import *



data = load_csv("./varySequenceLength/metrics/exp_varySequenceLength.csv")
replace_attr(data, "sequence", int)
replace_attr(data, "loss", float)
replace_attr(data, "time", float)
replace_attr(data, "dataSet", str)
replace_attr(data, "max_len", lambda x: "length:" + str(x))

_data = map(dict, data)
p = ggplot(_data, aes(x="time", y="loss", color="max_len", group="max_len", shape = "max_len"))
p += geom_point(size=0.3, alpha=0.3) + geom_path(size=0.3, alpha=0.6)
p += facet_grid("~dataSet", scales=esc("free_y"))
p += axis_labels("Time (s)", "")
p += legend_bottom
ggsave("exp_varySequenceLength.png", p, width=10, height=6, scale=.8)

data = load_csv("./varyDictSize/metrics/exp_varyDictSize.csv")
replace_attr(data, "sequence", int)
replace_attr(data, "loss", float)
replace_attr(data, "time", float)
replace_attr(data, "dataSet", str)
replace_attr(data, "vocab_size", lambda x: "size:" + str(x))

_data = map(dict, data)
p = ggplot(_data, aes(x="time", y="loss", color="vocab_size", group="vocab_size", shape = "vocab_size"))
p += geom_point(size=0.3, alpha=0.3) + geom_path(size=0.3, alpha=0.6)
p += facet_grid("~dataSet", scales=esc("free_y"))
p += axis_labels("Time (s)", "")
p += legend_bottom
ggsave("exp_varyDictSize.png", p, width=10, height=6, scale=.8)

data = load_csv("./varyTrainingSize/metrics/exp_varyTrainingSize.csv")
replace_attr(data, "sequence", int)
replace_attr(data, "loss", float)
replace_attr(data, "time", float)
replace_attr(data, "dataSet", str)
replace_attr(data, "num_examples", lambda x: "training:" + str(x))

_data = map(dict, data)
p = ggplot(_data, aes(x="time", y="loss", color="num_examples", group="num_examples", shape = "num_examples"))
p += geom_point(size=0.3, alpha=0.3) + geom_path(size=0.3, alpha=0.6)
p += facet_grid("~dataSet", scales=esc("free_y"))
p += axis_labels("Time (s)", "")
p += legend_bottom
ggsave("exp_varyTrainingSize.png", p, width=10, height=6, scale=.8)

data = load_csv("./varySpecialSize/metrics/exp_varySpecialSize.csv")
replace_attr(data, "sequence", int)
replace_attr(data, "loss", float)
replace_attr(data, "time", float)
replace_attr(data, "dataSet", str)
replace_attr(data, "special_num", lambda x: "special:" + str(x))

_data = map(dict, data)
p = ggplot(_data, aes(x="time", y="loss", color="special_num", group="special_num", shape = "special_num"))
p += geom_point(size=0.3, alpha=0.3) + geom_path(size=0.3, alpha=0.6)
p += facet_grid("~dataSet", scales=esc("free_y"))
p += axis_labels("Time (s)", "")
p += legend_bottom
ggsave("exp_varySpecialSize.png", p, width=10, height=6, scale=.8)