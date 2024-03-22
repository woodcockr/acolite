## def query
## queries CDSE for scenes
## written by Quinten Vanhellemont, RBINS
## 2023-09-12
## modifications: 2023-09-20 (QV) moved roi_wkt to function

def query(scene = None, collection = None, product = None,
               start_date = None, end_date = None,  roi = None,
               cloud_cover = None, tile = None, ## S2
               bright_cover = None, timeliness = None, ## S3
               verbosity = 1,
               max_results = 1000, odata_url = None, attributes = False):

    import os, requests, json
    import dateutil.parser, datetime
    from osgeo import ogr,osr,gdal
    import acolite as ac

    ## get odata url from ACOLITE config
    if odata_url is None: odata_url = ac.config['CDSE_odata']

    ## get collection info from scene
    if scene is not None:
        if ('MSIL1C' in scene) | (scene[0:3] in ['S2A', 'S2B']):
            collection = "SENTINEL-2"
            product = "S2MSI1C" ## S2MSI1C for Level 1 MSI data

        if ('SEN3' in scene) | (scene[0:3] in ['S3A', 'S3B']):
            collection = "SENTINEL-3"
            product = scene[4:15] ## OL_1_EFR___ for Level 1 full resolution OLCI data

    ## if scene is not given we need at least collection and product info
    if (collection is None):
        print('Please provide collection (SENTINEL-2 or SENTINEL-3) for query without scene identifier')
        return

    if (product is None):
        ## use defaults
        if collection == 'SENTINEL-2':
            product = "S2MSI1C"
            if verbosity > 0: print('Using default product {} for {}'.format(product, collection))
        if collection == 'SENTINEL-3':
            product = "OL_1_EFR___"
            if verbosity > 0: print('Using default product {} for {}'.format(product, collection))
        if (product is None):
            print('Please provide product (e.g. S2MSI1C or OL_1_EFR___)  for query without scene identifier')
            return

    ## determine WKT from provided ROI
    wkt = None
    if (roi is not None): wkt = ac.shared.roi_wkt(roi)
    if wkt is not None:
        if verbosity > 1: print('Using WKT for query: {}'.format(wkt))

    ## create query items
    query_list = []
    if scene is not None:
        ## remove extension for querying
        scene_ = '{}'.format(scene)
        scene_ = scene_.replace('.SAFE', '')
        scene_ = scene_.replace('.SEN3', '')
        query_list.append(f"contains(Name,'{scene_}')")

    if collection is not None:
        query_list.append(f"Collection/Name eq '{collection}'")

    if start_date is not None:
        sdate = dateutil.parser.parse(start_date)
        query_list.append(f"ContentDate/Start gt '{sdate.isoformat()}'")

    if end_date is not None:
        edate = dateutil.parser.parse(end_date)
        edate += datetime.timedelta(days=1) ## add one day to include end date data
        query_list.append(f"ContentDate/Start lt '{edate.isoformat()}'")

    ## OData intersection query
    if wkt is not None:
        query_list.append(f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt}')")

    ## attribute queries
    if product is not None:
        query_list.append(f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq '{product}')")
    if cloud_cover is not None:
        query_list.append(f"Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value lt {cloud_cover})")
    if tile is not None:
        query_list.append(f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'tileId' and att/OData.CSC.StringAttribute/Value eq '{tile}')")
    if bright_cover is not None:
        query_list.append(f"Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'brightCover' and att/OData.CSC.DoubleAttribute/Value lt {bright_cover})")
    if timeliness is not None:
        if timeliness in ['NR', 'NT', 'ST']:
            query_list.append(f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'timeliness' and att/OData.CSC.StringAttribute/Value eq '{timeliness}')")

    ## create query url
    query = f"{odata_url}/Products?$filter={' and '.join(query_list)}"

    ## add top # otherwise run through pages
    if max_results is not None:
        if (max_results>0) & (max_results<=1000): query += '&$top={}'.format(max_results)

    ## also return attributes
    if attributes:
        query+='&$expand=Attributes'
        atts = []

    ## print clickable url
    if verbosity > 1: print(query.replace(' ', "%20"))

    ## find download urls and scenes
    urls = []
    scenes = []

    ## query url
    response = requests.get(query)
    results = response.json()

    if 'value' in results:
        if verbosity > 1: print("Found {} scenes".format(len(results['value'])))
        for v in results['value']:
            if verbosity > 2: print(v['Id'], v['Name'])
            url = f"{odata_url}/Products({v['Id']})/$value"
            urls.append(url)
            scenes.append(v['Name'])
            if attributes: atts.append(v['Attributes'])

    ## paginate results
    while '@odata.nextLink' in results:
        response = requests.get(results['@odata.nextLink'])
        results = response.json()
        if verbosity > 1: print("Found {} more scenes".format(len(results['value'])))
        for v in results['value']:
            if verbosity > 2: print(v['Id'], v['Name'])
            url = f"{odata_url}/Products({v['Id']})/$value"
            urls.append(url)
            scenes.append(v['Name'])
            if attributes: atts.append(v['Attributes'])

    if verbosity > 0: print('Found {} total scenes'.format(len(urls)))
    if attributes: return(urls, scenes, atts)
    return(urls, scenes)
