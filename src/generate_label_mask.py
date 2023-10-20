'''
    Generate binary label mask for input image from geojson file
'''

import os
import numpy as np

def load_vertices(geojson_path, class_name='EpithelialCell'):
    '''
    
    Return a list of vertices for each polygon in the geojson file
    Each polygon is np.array of shape (n, 2), where n is the number of vertices

    '''
    import json
    with open(geojson_path) as f:
        data = json.load(f)
    
    feature_collection = data['features']
    epithelial_cells = list(filter(
        lambda region: region["properties"]["classification"]["name"] == class_name, feature_collection))
    print('Found %d %s in %s'%(len(epithelial_cells), class_name, geojson_path))

    verts_list = []
    for cell in epithelial_cells:
        verts = cell["geometry"]["coordinates"]
        verts_list.append(np.array(verts).squeeze(0))

    return verts_list

if __name__ == '__main__':
    label_file = '../raw_data/A28-87_CP_lvl1_HandE _1_Merged_RAW_ch00.tif_Annotations.json'
    verts_list = load_vertices(label_file, class_name='EpithelialCell')
    print('%d cells labeled: \n'%(len(verts_list)), verts_list[0].shape)

    dx_max, dy_max = 0, 0
    for cell in verts_list:
        x_max = np.max(cell[:, 0])
        y_max = np.max(cell[:, 1])

        x_min = np.min(cell[:, 0])
        y_min = np.min(cell[:, 1])

        dx = x_max - x_min
        dy = y_max - y_min

        dx_max = max(dx_max, dx)
        dy_max = max(dy_max, dy)
    
    print('Max dx: %d, Max dy: %d'%(dx_max, dy_max))
        
    


