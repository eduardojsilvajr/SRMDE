import geopandas as gpd
import os, math 
from osgeo import gdal
import pathlib, argparse
from typing import List, Optional, Literal


## Functions
# def snap_bounds(bounds, res):
#     minx, miny, maxx, maxy = bounds
#     # piso no min, teto no max para garantir cobertura total
#     minx_s = math.floor(minx / res) * res
#     miny_s = math.floor(miny / res) * res
#     maxx_s = math.ceil (maxx / res) * res
#     maxy_s = math.ceil (maxy / res) * res
#     return (minx_s, miny_s, maxx_s, maxy_s)

def call_gdalWarp(destName, sourceDst, res, cutlineDs, bounds):
    """call gdal.warp to cut the DEM and aligne it with the cutline dataset

    Args:
        destName (_type_): _description_
        sourceDst (_type_): _description_
        res (_type_): _description_
        cutlineDs (_type_): _description_
        bounds (_type_): _description_
    """
    gdal.Warp(
        destNameOrDestDS=destName,
        srcDSOrSrcDSTab=sourceDst,
        format='GTiff', 
        xRes=res, yRes=res,
        # outputBounds=bounds,
        # targetAlignedPixels=True,
        dstSRS = 'EPSG:4326',
        resampleAlg='near',
        cutlineDSName=cutlineDs,
        # cutlineLayerSRS = 'EPSG:4326',
        cropToCutline=True, 
        dstNodata=-9999,
        outputType=gdal.GDT_Float32 
)
    
def lista_tifs(diretorio: str, dem_type:Literal['dsm', 'dtm'])-> List:
    caminhos: List[str] = []
    for pasta_raiz, _, arquivos in os.walk(diretorio):
        for arquivo in arquivos:
            # Verifica se o arquivo é .tif e contém 'dsm' no nome (case-insensitive)
            if arquivo.lower().endswith('.tif') and dem_type in arquivo.lower():
                caminho_completo = os.path.join(pasta_raiz, arquivo)
                caminhos.append(caminho_completo)
    return caminhos
        


def main(args):
    
    bd_mestrado_path = r'D:\Documentos\OneDrive\Documentos\Mestrado\Dados\BD_mestrado.gpkg'
    blocos_ram = gpd.read_file(filename=bd_mestrado_path, layer='blocos_radiografia')
        
    root_path = r"C:\Users\Eduardo JR\Fast\SRIU\\"
    region = gpd.read_file(root_path+'bd_srmde.gpkg', layer='esrgan_finetuning_50k')
    
    phase = gpd.read_file(root_path+'bd_srmde.gpkg', layer=f'esrgan_finetuning_50k_{args.phase}')
    
    phase_filter = gpd.overlay(region, phase, how='intersection')

    region_mi_bloco = gpd.overlay(phase_filter, blocos_ram, how='intersection')
    #pegar os caminhos correspondentes dos MDE em region


    # Lista para armazenar os arquivos encontrados
    arquivos_tif = lista_tifs( diretorio=r"D:\RAM", dem_type=args.dem_type)

    res_30m_4326 = 0.000269494585235856472
    res_hr = res_30m_4326/args.zoom
    snaped_bound_15m = None # snap_bounds(region.total_bounds, res_30m_4326)

    for mi in region_mi_bloco.itertuples():
        bloco_ram_path = [s for s in arquivos_tif if mi.Nome.lower() in s]
        
        mi_aoi = gpd.GeoDataFrame(geometry=[mi.geometry], crs='EPSG:4326')
        cutline_ds = root_path+"region_merged.shp"
        mi_aoi.to_file(filename=cutline_ds, encoding='utf-8')
        output_file_hr = root_path + f'zoom_{str(args.zoom)}\\{args.phase}\\ram_{mi.MI}_{args.dem_type}.tif'  #'\\ram_finetuning_15m.tif'
        print(f'cuting HR {mi.MI}')
        call_gdalWarp(destName=output_file_hr, sourceDst=bloco_ram_path, 
                res=res_hr, cutlineDs=cutline_ds, bounds=snaped_bound_15m)
        
        print(f'cuting LR {mi.MI}')
        output_file_lr = root_path + f'zoom_{str(args.zoom)}\\{args.phase}\\cop_{mi.MI}.tif'  #'\\ram_finetuning_15m.tif'
        lr_ds_path = "C:\\Users\\Eduardo JR\\Fast\\SRIU\\COP_FineTuning.tif"
        call_gdalWarp(destName=output_file_lr, sourceDst=lr_ds_path, 
                res=res_30m_4326, cutlineDs=cutline_ds, bounds=snaped_bound_15m)
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=' prepare DEM data')
    parser.add_argument('--zoom',
        type= int,
        help= 'zoom between HR and LR dataset',
        default= 4
    )
    parser.add_argument('--dem_type',
        type= str,
        help='dsm or dtm',
        default= 'dsm'  
    )
    parser.add_argument('--phase',
        type= str,
        help='train, val or test',
        default= 'test'   
    )
    
    args = parser.parse_args()
    
    main(args)


# paths_filtrado = []
# for nome in ['6869W0001N'] : # region.Nome  ['6869W0001N'] - predict :
#     paths_filtrado.append([arq for arq in arquivos_dsm_tif if nome.lower() in arq.lower()][0])


# region_merged = unary_union(region.geometry)
# gdf_merged = gpd.GeoDataFrame(geometry=[region_merged], crs='EPSG:4326')
# output_path = root_path+"region_merged.shp"
# gdf_merged.to_file(output_path, encoding='utf-8')

# res_30m_4326 = 0.000269494585235856472
# res_15m = res_30m_4326/2.0
# snaped_bound_15m = None # snap_bounds(region.total_bounds, res_30m_4326)

# output_file = root_path + '\\ram_predict_30m.tif'  #'\\ram_finetuning_15m.tif'
# call_gdalWarp(destName=output_file, sourceDst=paths_filtrado, 
#               res=res_15m, cutlineDs=output_path, bounds=snaped_bound_15m)
# print('RAM')
# for ds in ['COP', 'SRTM']:
#     call_gdalWarp(destName=f"C:\\Users\\Eduardo JR\\Fast\\SRIU\\{ds}_predict_res30m.tif", 
#               sourceDst=f"C:\\Users\\Eduardo JR\\Fast\\SRIU\\{ds}_predict.tif", 
#               res=res_30m_4326, cutlineDs=output_path, bounds=snaped_bound_15m)
#     print(ds)

# a melhor opção para alinhar os DEM está sendo a função que eu já usava,
# essa função do snap só aumentou o erro.