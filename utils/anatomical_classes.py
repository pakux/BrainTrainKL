import seaborn as sns

set2_colors = sns.color_palette("Set2", 8)

ANATOMICAL_CATEGORIES = {
    "Frontal": {
        "color": set2_colors[0],
        "keywords": [
            "Frontal_Sup",
            "Frontal_Mid",
            "Frontal_Inf",
            "Frontal_Med_Orb",
            "Precentral",
            "Rolandic",
            "Supp_Motor_Area",
            "Olfactory",
            "Rectus",
            "OFCant",
            "OFClat",
            "OFCmed",
            "OFCpost",
        ],
    },
    "Parietal": {
        "color": set2_colors[1],
        "keywords": [
            "Parietal_Sup",
            "Parietal_Inf",
            "SupraMarginal",
            "Angular",
            "Precuneus",
            "Postcentral",
            "Paracentral_Lobule",
        ],
    },
    "Temporal": {
        "color": set2_colors[2],
        "keywords": [
            "Temporal_Sup",
            "Temporal_Pole_Sup",
            "Temporal_Mid",
            "Temporal_Pole_Mid",
            "Temporal_Inf",
            "Heschl",
            "Fusiform",
            "Insula",
        ],
    },
    "Occipital": {
        "color": set2_colors[3],
        "keywords": [
            "Occipital_Sup",
            "Occipital_Mid",
            "Occipital_Inf",
            "Cuneus",
            "Lingual",
            "Calcarine",
        ],
    },
    "Limbic": {
        "color": set2_colors[4],
        "keywords": [
            "Cingulate_Ant",
            "Cingulate_Mid",
            "Cingulate_Post",
            "Hippocampus",
            "ParaHippocampal",
            "Amygdala",
            "ACC_pre",
            "ACC_sub",
            "ACC_sup",
        ],
    },
    "Brain Stem": {
        "color": set2_colors[5],
        "keywords": ["LC_", "Raphe_", "VTA_", "Red_N", "SN_", "Region_"],
    },
    "Basal Ganglia": {
        "color": set2_colors[6],
        "keywords": ["Thal_", "Caudate", "Putamen", "Pallidum", "N_Acc", "Thal"],
    },
    "Cerebellum": {
        "color": set2_colors[7],
        "keywords": [
            "Cerebellum_Crus1",
            "Cerebellum_Crus2",
            "Cerebellum_3",
            "Cerebellum_4_5",
            "Cerebellum_6",
            "Cerebellum_7b",
            "Cerebellum_8",
            "Cerebellum_9",
            "Cerebellum_10",
            "Vermis_1_2",
            "Vermis_3",
            "Vermis_4_5",
            "Vermis_6",
            "Vermis_7",
            "Vermis_8",
            "Vermis_9",
            "Vermis_10",
        ],
    },
}
