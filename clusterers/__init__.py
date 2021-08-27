from clusterers import (base_clusterer, random_labels, single_label,
                        scan_selflabel_guide,
                        toy_guide_clusterer)

clusterer_dict = {
    'supervised': base_clusterer.BaseClusterer,
    'random_labels': random_labels.Clusterer,
    'single_label': single_label.Clusterer,
    'scan_guide': scan_selflabel_guide.ScanSelflabelGuide,
    'toy_guide_clusterer': toy_guide_clusterer.ToyGuideClusterer
}

#### Use below for toy examples
# from clusterers import (base_clusterer, random_labels,
#                         toy_guide_clusterer)

# clusterer_dict = {
#     'supervised': base_clusterer.BaseClusterer,
#     'random_labels': random_labels.Clusterer,
#     'toy_guide_clusterer': toy_guide_clusterer.ToyGuideClusterer,
# }
