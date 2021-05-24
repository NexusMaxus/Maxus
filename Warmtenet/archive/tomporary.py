def spatial_weights(self):
    """
    Combines the rasters of s1 and s2 raster data for the field to
    a biomass proxy raster. It does so by using weights and quality indicators
    and timeseries

    Returns
    -------

    """
    client = storage.Client()
    bucket = client.get_bucket(self.bucket_name)

    s1_raster_stack_list = []
    if self.s1obj.source == 'GOOGLE':
        for platform, orbit, orbitnumber in self.s1obj.combis:
            bucket_addr = '{}/{}/s1_{}_to_ndvi_one_relation/{}/{}/{}/'.format(
                self.s1obj.bucket_addr, self.s1obj.subfolder,
                self.s1obj.in_res, platform, orbit, orbitnumber)
            bucket_name, prefix = gutl.split_bucket_prefix(bucket_addr)
            filelist = sorted(gutl.get_filelist_from_gcloud(self.bucket_addr, prefix, storage_client=client,
                                                            bucket=bucket, mount_point=self.bucket_addr))
            vsigs_list = [gutl.gs_address_to_gdal_fname(file) for file in filelist]
            vsigs_series = pd.Series(data=vsigs_list, index=extract_dates(vsigs_list)).sort_index()[
                           self.s1obj.start_date:self.s1obj.end_date]
            s1_raster_stack = self.s1obj.get_raster_series_for_field(vsigs_series=vsigs_series, ogr_geom=self.ogr_geom,
                                                                     min_coverage=0)

            bucket_addr = '{}/{}/s1_{}/{}/{}/{}/vhvv/'.format(
                self.s1obj.bucket_addr, self.s1obj.subfolder,
                self.s1obj.in_res, platform, orbit, orbitnumber)
            bucket_name, prefix = gutl.split_bucket_prefix(bucket_addr)
            filelist = sorted(gutl.get_filelist_from_gcloud(self.bucket_addr, prefix, storage_client=client,
                                                            bucket=bucket, mount_point=self.bucket_addr))
            vsigs_list = [gutl.gs_address_to_gdal_fname(file) for file in filelist]
            vsigs_series = pd.Series(data=vsigs_list, index=extract_dates(vsigs_list)).sort_index()[
                           self.s1obj.start_date:self.s1obj.end_date]

            s1_raster_stack_full_cov = self.s1obj.get_raster_series_for_field(vsigs_series=vsigs_series,
                                                                              ogr_geom=self.ogr_geom,
                                                                              min_coverage=1)
            if len(s1_raster_stack_full_cov) > 0:
                s1_raster_stack_filtered = s1_raster_stack[s1_raster_stack_full_cov.index]
                s1_raster_stack_list.append(s1_raster_stack_filtered)

    elif self.s1obj.source == 'EODC':
        for orbit in self.s1obj.combis:
            bucket_addr = '{}/{}/s1_{}_to_ndvi_one_relation_yoann/{}/'.format(
                self.bucket_addr, self.s1obj.subfolder, self.s1obj.in_res, orbit)
            bucket_name, prefix = gutl.split_bucket_prefix(bucket_addr)
            filelist = sorted(gutl.get_filelist_from_gcloud(self.s1obj.bucket_addr, prefix, storage_client=client,
                                                            bucket=bucket, mount_point=self.bucket_addr))

            vsigs_list = [gutl.gs_address_to_gdal_fname(file) for file in filelist]
            date_list = [x.split('/')[-1] for x in vsigs_list]
            vsigs_series = pd.Series(data=vsigs_list, index=extract_dates(date_list)).sort_index()
            s1_raster_stack = self.s1obj.get_raster_series_for_field(vsigs_series=vsigs_series,
                                                                     ogr_geom=self.ogr_geom,
                                                                     min_coverage=0.998)
            s1_raster_stack_list.append(s1_raster_stack)

    s1_raster_stack = pd.concat(s1_raster_stack_list).dropna().sort_index()

    bucket_addr = '{}/{}/s2_{}/level{}/ndvi_vds/'.format(self.s2obj.bucket_addr, self.s2obj.subfolder,
                                                         self.s2obj.in_res, self.s2obj.level)
    bucket_name, prefix = gutl.split_bucket_prefix(bucket_addr)
    filelist = sorted(gutl.get_filelist_from_gcloud(bucket_addr, prefix, storage_client=client,
                                                    bucket=bucket, mount_point='gs://bendvi'))
    ndvi_tiff_list = [gutl.gs_address_to_gdal_fname(file) for file in filelist]
    ndvi_series = pd.Series(data=ndvi_tiff_list, index=extract_dates(ndvi_tiff_list)
                            )[self.s2obj.start_date:self.s2obj.end_date]

    bucket_addr = '{}/{}/s2_{}/level{}/cloudandsnowmask/'.format(self.s2obj.bucket_addr, self.s2obj.subfolder,
                                                                 self.s2obj.in_res, self.s2obj.level)
    bucket_name, prefix = gutl.split_bucket_prefix(bucket_addr)
    filelist = sorted(gutl.get_filelist_from_gcloud(bucket_addr, prefix, storage_client=client,
                                                    bucket=bucket, mount_point='gs://bendvi'))
    tiff_list = [gutl.gs_address_to_gdal_fname(file) for file in filelist]
    cloud_series = pd.Series(data=tiff_list, index=extract_dates(tiff_list)
                             )[self.s2obj.start_date:self.s2obj.end_date]

    s2_cloud_snow_mask = self.s2obj.get_raster_series_for_field(vsigs_series=cloud_series,
                                                                ogr_geom=self.ogr_geom, min_coverage=0)

    clear_terrain_ratio = []
    for mask in s2_cloud_snow_mask:
        cloudy_pixels = mask.compressed().sum()
        pixels = len(mask.compressed())
        clear_terrain_ratio.append(1 - cloudy_pixels / pixels)

    clear_terrain_series = pd.Series(clear_terrain_ratio, index=s2_cloud_snow_mask.index)

    coverage = 1
    non_cloudy_series = clear_terrain_series[clear_terrain_series >= coverage]
    while isinstance(non_cloudy_series, float):
        coverage -= 0.001
        non_cloudy_series = clear_terrain_series[clear_terrain_series >= coverage]

    while not ((len(non_cloudy_series) > self.ps.min_obs_full_cov_py) and (len(non_cloudy_series[:'2018-01-01']) > 0)):
        coverage -= 0.001
        non_cloudy_series = clear_terrain_series[clear_terrain_series >= coverage]

    if coverage < 0.8:
        print('CHECK FIELD {} ON COVERAGE {}'.format(self.geom_id, coverage))

    s2_raster_stack = self.s2obj.get_raster_series_for_field(vsigs_series=ndvi_series,
                                                             ogr_geom=self.ogr_geom, min_coverage=0)

    s2_raster_stack_full_cov = s2_raster_stack[non_cloudy_series.index].dropna()

    s2_ts_full_cov, s2_qf_ts_full = self.s2obj.get_ts_for_shape(ogr_geom=self.ogr_geom, geom_id=self.geom_id,
                                                                type_s2=self.ps.type_s2,
                                                                end_date=self.s2obj.end_date,
                                                                raster_series=s2_raster_stack_full_cov)

    # get geotransform
    cut_fn = './data/cut_file_ref_{}.tif'.format(self.geom_id)

    geotiff_roi_cut(gdal_ds_name=ndvi_tiff_list[0],
                    ogr_geom=self.ogr_geom,
                    output_fname=cut_fn,
                    crop_to_geom=True,
                    output_format='GTIFF')
    gt = gdal.Open(cut_fn).GetGeoTransform()

    for date, value in self.beta_ts.dropna().items():
        if (date >= self.s1_qf_ts_scaled.index[0]) and (date >= s2_raster_stack_full_cov.index[0]) and \
                (date >= s1_raster_stack.index[0]):
            s1_rasters = s1_raster_stack[:date].iloc[-self.ps.obs_for_s1_comp:]

            s1_weights = []
            for s1_raster in s1_rasters:
                s1_base = s1_raster.base
                non_field_mask = s1_raster.mask
                s1_base[non_field_mask] = np.NaN
                s1_ratio_single = s1_base / np.nanmean(s1_base)
                s1_ratio_single[s1_ratio_single * value > 1] = 1 / value
                s1_weights.append(s1_ratio_single)

            s1_ratio = np.array(s1_weights).mean(axis=0)

            s2_raster = s2_raster_stack_full_cov[:date].iloc[-1]
            non_field_mask = s2_raster.mask
            field_base = s2_raster.base
            field_base[non_field_mask] = np.NaN
            field_base[field_base == self.s2obj.no_data_value_clouds] = np.NaN
            s2_ratio = field_base / np.nanmean(field_base)

            s1_factor = (s1_ratio * self.ps.ws_s1) / (self.s1_qf_ts_scaled.loc[date] + s2_qf_ts_full.loc[date]) * \
                        (self.ps.ws_s1 + self.ps.ws_s2)
            s2_factor = (s2_ratio * self.ps.ws_s2) / (self.s1_qf_ts_scaled.loc[date] + s2_qf_ts_full.loc[date]) * \
                        (self.ps.ws_s1 + self.ps.ws_s2)
            total_weights = (s1_ratio * s1_factor + s2_ratio * s2_factor) / (s1_factor + s2_factor)

            tws = total_weights / np.nanmean(total_weights)

            for iteration in range(self.ps.iterations_weighing):
                tws[tws * value > 1] = 1 / value
                tws = tws / np.nanmean(tws)

            final_beta = tws * value
            final_beta[final_beta > 1] = 1

            tmp_fn = './data/cut_beta_file_ref_{}_out.tif'.format(self.geom_id)
            write_geotiff(tmp_fn, final_beta, gt)

            fn_bucket = '{}/{}/fusion/delta11_yoann/fields/{}/{}.tif'.format(
                self.s2obj.bucket_addr, self.s2obj.subfolder,
                self.geom_id, date.strftime('%Y-%m-%d'))
            gutl.move_to_gcs_or_local(tmp_fn,
                                      fn_bucket,
                                      storage_client=self.client,
                                      bucket=self.bucket)

    return