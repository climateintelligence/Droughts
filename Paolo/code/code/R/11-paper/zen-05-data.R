# write data for zenodo

library(data.table)
library(magrittr)
library(fs)
library(readxl)


# read data ---------------------------------------------------------------



path_out <- "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/zenodo-data/"
path_out_temp <- path(path_out, "temp")
dir_create(path_out_temp)

dat_share <- read_excel("data/00_summary.xlsx", range = "shareable!A2:E18")
setDT(dat_share)

dat_meta <- unique(rbind(
  readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1787hn_1879hs-1960/r-data/meta_long_HN_HS.rds"),
  readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1961-2020/r-data/meta_long_HN_HS.rds")
))


dat_hshn_01_orig <- rbind(
  readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1787hn_1879hs-1960/r-data/data_long_HN_HS.rds"),
  readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1961-2020/r-data/data_long_HN_HS.rds")
)


dat_hshn_02_qc <- rbind(
  readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/rds/1787hn_1879hs-1960/data_long_HN_HS.rds"),
  readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/rds/1961-2020/data_long_HN_HS.rds")
)
setnames(dat_hshn_02_qc, c("HN", "HS"), c("HN_after_qc", "HS_after_qc"))

dat_hshn_03_gapfill <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/04_GAPFILL/rds/gapfill-01-all.rds")
setnames(dat_hshn_03_gapfill, "HS", "HS_after_gapfill")
dat_hshn_03_gapfill[, HS_fillcode := NULL]



# meta --------------------------------------------------------------------


dat_meta_year_hn <- dat_hshn_01_orig[!is.na(HN),
                                     .(HN_year_start = min(year(Date)),
                                       HN_year_end = max(year(Date))),
                                     .(Name)]

dat_meta_year_hs <- dat_hshn_01_orig[!is.na(HS),
                                     .(HS_year_start = min(year(Date)),
                                       HS_year_end = max(year(Date))),
                                     .(Name)]
  

dat_meta %>% 
  merge(dat_meta_year_hn, all.x = T) %>% 
  merge(dat_meta_year_hs, all.x = T) -> dat_meta_out

sf_meta_out <- sf::st_as_sf(dat_meta_out,
                            coords = c("Longitude", "Latitude"),
                            crs = 4326)
mv <- mapview::mapview(sf_meta_out, zcol = "Elevation")
mapview::mapshot(mv, 
                 url = path(path_out, "meta_interactive_map", ext = "html"),
                 selfcontained = T)
# remove temp files
dir_delete(path(path_out, "meta_interactive_map_files"))


setorder(dat_meta_out, Provider, Name)
setcolorder(dat_meta_out, "Provider")
dat_meta_out

fwrite(dat_meta_out,
       path(path_out, "meta_all", ext = "csv"))

# daily -------------------------------------------------------------------

# HN, HN_after_qc, HS, HS_after_qc, HS_after_gapfill

prov_daily <- dat_share[daily == "yes", Code]
print(prov_daily)

for(i_prov in prov_daily){
  
  i_stns <- dat_meta[Provider == i_prov, Name]
  
  dat_hshn_01_orig[Name %in% i_stns] %>% 
    merge(dat_hshn_02_qc[Name %in% i_stns]) %>% 
    merge(dat_hshn_03_gapfill[Name %in% i_stns]) -> dat_out
  
  setcolorder(dat_out, "Provider")
  setorder(dat_out, Provider, Name, Date)
  
  
  fn_csv <- path(path_out_temp, paste0("data_daily_", i_prov), ext = "csv")
  fn_zip <- path(path_out, paste0("data_daily_", i_prov), ext = "zip")
  
  fwrite(dat_out, fn_csv)
  zip::zip(fn_zip, fn_csv, mode = "cherry-pick")
  
  
  
}


# monthly -----------------------------------------------------------------

dat_monthly_01_orig <- rbind(
  readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/05_MONTHLY/rds/1787hn_1879hs-1960/data_long_HN_HS.rds"),
  readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/05_MONTHLY/rds/1961-2020/data_long_HN_HS.rds")
)
dat_monthly_02_gapfill <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/05_MONTHLY/rds/1961-2020_gapfilled_HS_sub/data_long_HS.rds")
setnames(dat_monthly_02_gapfill, c("HS", "frac_gapfilled"), c("HS_gapfill", "HS_gapfill_frac"))

# HN, HS, HS_gapfill, HS_gapfill_frac

prov_monthly <- dat_share[monthly == "yes", Code]
print(prov_monthly) 

for(i_prov in prov_monthly){
  
  i_stns <- dat_meta[Provider == i_prov, Name]
  
  dat_monthly_01_orig[Name %in% i_stns] %>% 
    merge(dat_monthly_02_gapfill[Name %in% i_stns], 
          by = c("Name", "year", "month"), all = T) -> dat_out
  
  setorder(dat_out, Name, year, month)
  
  
  fn_csv <- path(path_out_temp, paste0("data_monthly_", i_prov), ext = "csv")
  fn_zip <- path(path_out, paste0("data_monthly_", i_prov), ext = "zip")
  
  fwrite(dat_out, fn_csv)
  zip::zip(fn_zip, fn_csv, mode = "cherry-pick")
  
  
  
}

