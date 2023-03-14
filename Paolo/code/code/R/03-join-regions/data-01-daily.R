# read in all original data

library(fs)
library(data.table)
library(magrittr)
library(readxl)
library(lubridate)

source("R/functions/safe_names.R")

# AT_HZB
# CH_METEOSWISS
# CH_SLF
# DE_DWD
# FR_METEOFRANCE
# IT_BZ
# IT_FVG
# IT_LOMBARDIA
# IT_PIEMONTE
# IT_SMI
# IT_TN
# IT_TN_TUM
# IT_VDA_AIBM
# IT_VDA_CF
# IT_VENETO
# SI_ARSO

provider_all <- c("AT_HZB", "CH_METEOSWISS", "CH_SLF", "DE_DWD", "FR_METEOFRANCE",
                  "IT_BZ", "IT_FVG", "IT_LOMBARDIA", "IT_PIEMONTE", "IT_SMI", 
                  "IT_TN", "IT_TN_TUM", "IT_VDA_AIBM", "IT_VDA_CF", 
                  "IT_VENETO", "SI_ARSO")



# meta read --------------------------------------------------------------------

# AT_HZB
meta_at <- fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/00_ORIGINAL_DATA/AT_HZB/list_sites.csv")

# CH_METEOSWISS
meta_ch_meteoswiss_hn <- fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/01_MERGING/METEOSWISS/list_mergedHN.csv") 
meta_ch_meteoswiss_hs <- fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/01_MERGING/METEOSWISS/list_mergedHS.csv") 

# CH_SLF
meta_ch_slf <- fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/00_ORIGINAL_DATA/CH_SLF/list_manual_sites.csv")

# DE_DWD
meta_de_hn <- fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/01_MERGING/GERMANY/list_mergedHN")  
meta_de_hs <- fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/01_MERGING/GERMANY/list_mergedHS")  

# FR_METEOFRANCE
meta_fr_hn <- fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/01_MERGING/SECOND_ROUND/METEOFRANCE/list_mergedHN.csv")  
meta_fr_hs <- fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/01_MERGING/SECOND_ROUND/METEOFRANCE/list_mergedHS.csv")  

# IT_BZ
meta_it_bz <- fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/00_ORIGINAL_DATA/BOLZANO/list_manual_sites.csv")  

# IT_FVG
meta_it_fvg <- fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/01_MERGING/FVG/list_mergedHS.csv")  

# IT_LOMBARDIA
read_excel("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/00_ORIGINAL_DATA/LOMBARDIA/list_of_manual_stations.xlsx") %>% 
  as.data.table() %>% 
  .[, .(Name, Longitude = Long, Latitude = Lat, Elevation = Alt)] -> meta_it_lombardia

# IT_PIEMONTE
meta_it_piemonte <- fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/00_ORIGINAL_DATA/PIEMONTE/list_manual_sites.csv")

# IT_SMI
meta_it_smi_hn <- fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/00_ORIGINAL_DATA/IT_SMI/list_sites_manual_HN.csv")
meta_it_smi_hs <- fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/01_MERGING/SECOND_ROUND/SMI/list_sites_manual_HS.csv")

# IT_TN
meta_it_tn <- fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/01_MERGING/SECOND_ROUND/TRENTINO/listMT.csv")

# IT_TN_TUM
meta_it_tn_tum <- fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/01_MERGING/SECOND_ROUND/TRENTINO/listTUM.csv",
                        skip = 1, col.names = names(meta_it_tn))

# IT_VDA_AIBM
meta_it_vda_aibm <- fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/01_MERGING/SECOND_ROUND/VDA/list_sites_HS_AIBM.csv")

# IT_VDA_CF
meta_it_vda_cf <- fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/01_MERGING/SECOND_ROUND/VDA/list_sites_HS_CF.csv",
                        encoding = "Latin-1")

# IT_VENETO
meta_it_veneto <- fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/00_ORIGINAL_DATA/VENETO/list_manual_sites_filledNAmanual.csv")

# SI_ARSO
meta_si_hn <- fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/01_MERGING/SLOVENIA/List_stations_HN.csv", 
                    encoding = "Latin-1")
meta_si_hs <- fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/01_MERGING/SLOVENIA/List_stations_HS.csv", 
                    encoding = "Latin-1")



# meta join ---------------------------------------------------------------

# prep
setnames(meta_it_fvg, c("Name", "Latitude", "Longitude", "Elevation"))

meta_it_vda_cf[, Name := safe_names(Name)]

meta_si_hn <- meta_si_hn[!is.na(ID)]
meta_si_hn[, stn_name_safe := stringi::stri_trans_totitle(safe_names(tolower(NAME)))]
meta_si_hn_out <- meta_si_hn[, .(Name = stn_name_safe, Longitude = LONG, Latitude = LAT, Elevation = ELE)]

meta_si_hs[, stn_name_safe := stringi::stri_trans_totitle(safe_names(tolower(NAME)))]
meta_si_hs_out <- meta_si_hs[, .(Name = stn_name_safe, Longitude = LONG, Latitude = LAT, Elevation = ELE)]
# meta_si_hs %>% .[, .N, stn_name_safe] %>% .[N > 1, stn_name_safe]

# join hn
l_meta_hn <- list(meta_at, meta_ch_meteoswiss_hn, meta_ch_slf, meta_de_hn,
                  meta_fr_hn, meta_it_bz, meta_it_fvg, meta_it_lombardia,
                  meta_it_piemonte, meta_it_smi_hn, meta_it_tn, meta_it_tn_tum,
                  meta_it_vda_aibm, meta_it_vda_cf, meta_it_veneto, meta_si_hn_out)
names(l_meta_hn) <- provider_all
lapply(l_meta_hn, names)

join_meta_hn <- rbindlist(l_meta_hn, use.names = T, fill = T, idcol = "provider")

# join hs
l_meta_hs <- list(meta_at, meta_ch_meteoswiss_hs, meta_ch_slf, meta_de_hs,
                  meta_fr_hs, meta_it_bz, meta_it_fvg, meta_it_lombardia,
                  meta_it_piemonte, meta_it_smi_hs, meta_it_tn, meta_it_tn_tum,
                  meta_it_vda_aibm, meta_it_vda_cf, meta_it_veneto, meta_si_hs_out)
names(l_meta_hs) <- provider_all
lapply(l_meta_hs, names)

join_meta_hs <- rbindlist(l_meta_hs, use.names = T, fill = T, idcol = "provider")




# data read ---------------------------------------------------------------

f_read_plus <- function(fn){
  dat <- fread(fn)
  dat2 <- melt(dat, id.vars = "Date", variable.factor = F)
  dat2[, Date := ymd(Date)]
  dat2
}

f_read_plus2 <- function(fn){
  line1 <- readLines(fn, 1)
  line1 %>% gsub('\\"', '', .) %>% strsplit(",") %>% .[[1]] -> colnam

  dat <- fread(fn, skip = 1)
  setnames(dat, colnam)
  dat2 <- melt(dat, id.vars = "Date", variable.factor = F)
  dat2[, Date := ymd(Date)]
  dat2
}




# AT_HZB
data_at_hs <- f_read_plus("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/00_ORIGINAL_DATA/AT_HZB/read_HS.csv")
data_at_hn <- f_read_plus("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/00_ORIGINAL_DATA/AT_HZB/read_HN.csv")

# CH_METEOSWISS
data_ch_meteoswiss_hn <- f_read_plus2("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/01_MERGING/METEOSWISS/mergedHN.csv") 
data_ch_meteoswiss_hs <- f_read_plus2("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/01_MERGING/METEOSWISS/mergedHS.csv") 

# CH_SLF
data_ch_slf_hn <- f_read_plus("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/00_ORIGINAL_DATA/CH_SLF/read_manualHN.csv")
data_ch_slf_hs <- f_read_plus("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/00_ORIGINAL_DATA/CH_SLF/read_manualHS.csv")

# DE_DWD
data_de_hn <- f_read_plus("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/01_MERGING/GERMANY/mergedHN.csv")  
data_de_hs <- f_read_plus("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/01_MERGING/GERMANY/mergedHS.csv")  

# FR_METEOFRANCE
data_fr_hn <- f_read_plus2("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/01_MERGING/SECOND_ROUND/METEOFRANCE/mergedHN.csv")  
data_fr_hs <- f_read_plus2("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/01_MERGING/SECOND_ROUND/METEOFRANCE/mergedHS.csv")  

# IT_BZ
data_it_bz_hn <- f_read_plus("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/00_ORIGINAL_DATA/BOLZANO/read_manualHN.csv")  
data_it_bz_hs <- f_read_plus("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/00_ORIGINAL_DATA/BOLZANO/read_manualHS.csv")  

# IT_FVG
fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/01_MERGING/FVG/merged_HN.csv") %>% 
  melt(id.vars = "Date", variable.factor = F) -> data_it_fvg_hn
data_it_fvg_hn[, Date := dmy(Date)]
data_it_fvg_hn[startsWith(as.character(value), "9"), table(value)]
data_it_fvg_hn[startsWith(as.character(value), "8"), table(value)]
data_it_fvg_hn[value == 999 | value == 899, value := 0] # traces of HN or traces of HN with rain
data_it_fvg_hn[value == 800 , value := 0] # rain on snow, no fresh snow
data_it_fvg_hn[value > 800, value := value - 800]
data_it_fvg_hn[value < 0, value := NA]
summary(data_it_fvg_hn)

fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/01_MERGING/FVG/merged_HS.csv") %>% 
  melt(id.vars = "Date", variable.factor = F) -> data_it_fvg_hs
data_it_fvg_hs[, Date := dmy(Date)]
data_it_fvg_hs[value >= 800 | value < 0, value := NA]
summary(data_it_fvg_hs)

# IT_LOMBARDIA
fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/00_ORIGINAL_DATA/LOMBARDIA/read_manual_HN.csv") %>% 
  melt(id.vars = "date", variable.factor = F) -> data_it_lombardia_hn
setnames(data_it_lombardia_hn, "date", "Date")
data_it_lombardia_hn[, Date := dmy(Date)]
data_it_lombardia_hn[startsWith(as.character(value), "9"), table(value)]
data_it_lombardia_hn[startsWith(as.character(value), "8"), table(value)]
data_it_lombardia_hn[value > 800, value := NA] # as it seems no AINEVA codes
summary(data_it_lombardia_hn)

fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/00_ORIGINAL_DATA/LOMBARDIA/read_manual_HS.csv") %>% 
  melt(id.vars = "date", variable.factor = F) -> data_it_lombardia_hs
setnames(data_it_lombardia_hs, "date", "Date")
data_it_lombardia_hs[, Date := dmy(Date)]
summary(data_it_lombardia_hs)

# IT_PIEMONTE
data_it_piemonte_hn <- f_read_plus("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/00_ORIGINAL_DATA/PIEMONTE/read_manualHN.csv")
data_it_piemonte_hs <- f_read_plus("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/00_ORIGINAL_DATA/PIEMONTE/read_manualHS.csv")

# IT_SMI
data_it_smi_hn <- f_read_plus("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/00_ORIGINAL_DATA/IT_SMI/read_manualHN.csv")
data_it_smi_hs <- f_read_plus("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/01_MERGING/SECOND_ROUND/SMI/read_manualHS.csv")

# IT_TN
data_it_tn_hn <- f_read_plus("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/01_MERGING/TRENTINO/mergedHN.csv")
data_it_tn_hs <- f_read_plus2("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/01_MERGING/SECOND_ROUND/TRENTINO/dataMT.csv")

# IT_TN_TUM
data_it_tn_tum_hs <- f_read_plus("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/01_MERGING/SECOND_ROUND/TRENTINO/dataTUM.csv")

# IT_VDA_CF
data_it_vda_cf_hs <- fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/01_MERGING/SECOND_ROUND/VDA/read_HS_CF.csv",
                           encoding = "Latin-1")
data_it_vda_cf_hs <- melt(data_it_vda_cf_hs, id.vars = "Date", variable.factor = F)
data_it_vda_cf_hs[, Date := ymd(Date)]
data_it_vda_cf_hs[, variable := safe_names(variable)]

# IT_VDA_AIBM
data_it_vda_aibm_hn <- f_read_plus("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/00_ORIGINAL_DATA/VDA_AIBM/read_HN.csv")
data_it_vda_aibm_hs <- f_read_plus("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/01_MERGING/SECOND_ROUND/VDA/read_HS_AIBM.csv")


# IT_VENETO
fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/00_ORIGINAL_DATA/VENETO/read_manualHN.csv",
      na.strings = c("NA", "///")) %>% 
  melt(id.vars = "Date", variable.factor = F) -> data_it_veneto_hn
data_it_veneto_hn[, Date := dmy(Date)]
summary(data_it_veneto_hn)

data_it_veneto_hs <- fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/01_MERGING/VENETO/read_manualHS_complete.csv")
i_century <- which(data_it_veneto_hs$Date == "01.01.00")
data_it_veneto_hs[1:(i_century-1), Date := paste0(substr(Date, 1, 6), "19", substr(Date, 7, 8))]
data_it_veneto_hs[i_century:.N, Date := paste0(substr(Date, 1, 6), "20", substr(Date, 7, 8))]
data_it_veneto_hs <- melt(data_it_veneto_hs, id.vars = "Date", variable.factor = F)
data_it_veneto_hs[, Date := dmy(Date)]
data_it_veneto_hs[value < 0, value := 0] # some weird single value
summary(data_it_veneto_hs)


# SI_ARSO
data_si_hn <- fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/01_MERGING/SLOVENIA/merged_HN.csv", header = T) 
setnames(data_si_hn, "date", "Date")
i_century <- which(data_si_hn$Date == "01.01.00")
data_si_hn[1:(i_century-1), Date := paste0(substr(Date, 1, 6), "19", substr(Date, 7, 8))]
data_si_hn[i_century:.N, Date := paste0(substr(Date, 1, 6), "20", substr(Date, 7, 8))]
data_si_hn <- melt(data_si_hn, id.vars = "Date", variable.factor = F)
data_si_hn[, Date := dmy(Date)]
data_si_hn[value == -1, value := 0] # days without snow
summary(data_si_hn)
data_si_hn[, variable := as.integer(variable)]
data_si_hn <- merge(data_si_hn, meta_si_hn[, .(variable = ID, stn_name_safe)])
data_si_hn[, variable := NULL]
setnames(data_si_hn, "stn_name_safe", "variable")

data_si_hs <- fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/01_MERGING/SLOVENIA/merged_HS.csv", header = T) 
setnames(data_si_hs, "date", "Date")
i_century <- which(data_si_hs$Date == "01.01.00")
data_si_hs[1:(i_century-1), Date := paste0(substr(Date, 1, 6), "19", substr(Date, 7, 8))]
data_si_hs[i_century:.N, Date := paste0(substr(Date, 1, 6), "20", substr(Date, 7, 8))]
data_si_hs <- melt(data_si_hs, id.vars = "Date", variable.factor = F)
data_si_hs[, Date := dmy(Date)]
data_si_hs[value == -1, value := 0] # days without snow
summary(data_si_hs)
data_si_hs[, variable := as.integer(variable)]
data_si_hs <- merge(data_si_hs, meta_si_hs[, .(variable = ID, stn_name_safe)])
data_si_hs[, variable := NULL]
setnames(data_si_hs, "stn_name_safe", "variable")


# data join ---------------------------------------------------------------


# join hn
l_data_hn <- list(data_at_hn, data_ch_meteoswiss_hn, data_ch_slf_hn, data_de_hn,
                  data_fr_hn, data_it_bz_hn, data_it_fvg_hn, data_it_lombardia_hn,
                  data_it_piemonte_hn, data_it_smi_hn, data_it_tn_hn, data_it_vda_aibm_hn,
                  data_it_veneto_hn, data_si_hn)
names(l_data_hn) <- provider_all[!provider_all %in% c("IT_TN_TUM", "IT_VDA_CF")]
lapply(l_data_hn, names)

join_data_hn <- rbindlist(l_data_hn, use.names = T, fill = T, idcol = "provider")


# join hs
l_data_hs <- list(data_at_hs, data_ch_meteoswiss_hs, data_ch_slf_hs, data_de_hs,
                  data_fr_hs, data_it_bz_hs, data_it_fvg_hs, data_it_lombardia_hs,
                  data_it_piemonte_hs, data_it_smi_hs, data_it_tn_hs, data_it_tn_tum_hs,
                  data_it_vda_aibm_hs, data_it_vda_cf_hs, data_it_veneto_hs, data_si_hs)
names(l_data_hs) <- provider_all
lapply(l_data_hs, names)

join_data_hs <- rbindlist(l_data_hs, use.names = T, fill = T, idcol = "provider")


# final checks ------------------------------------------------------------

# meta & data for VDA_CF correspond
# stns_data <- data_it_vda_cf_hs[, .(Name = unique(variable), in_data = "yes")]
# merge(meta_it_vda_cf, stns_data, all = T)
# join_data_hn[variable == "Bioz_Place_Moulin"]
# join_data_hs[variable == "Bioz_Place_Moulin", variable := "Bionaz_Place_Moulin"]
# data_it_vda_cf_hs[variable == "Bioz_Place_Moulin", variable := "Bionaz_Place_Moulin"]

# SMI not duplicates in PIEMONTE & VDA (not in second round)
# smi_duplicated <- fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/01_MERGING/SMI/duplicates_smi_piemonte.txt")
# meta_it_smi_hs
# join_meta_hn <- join_meta_hn[!(provider == "IT_PIEMONTE" & Name %in% smi_duplicated$Piedmont)]
# join_meta_hs <- join_meta_hs[!(provider == "IT_PIEMONTE" & Name %in% smi_duplicated$Piedmont)]
# join_data_hn <- join_data_hn[!(provider == "IT_PIEMONTE" & variable %in% smi_duplicated$Piedmont)]
# join_data_hs <- join_data_hs[!(provider == "IT_PIEMONTE" & variable %in% smi_duplicated$Piedmont)]

                        
# meta names unique (esp SLF & MSWISS)
join_meta_hn[, .N, .(provider, Name)] %>% summary
join_meta_hs[, .N, .(provider, Name)] %>% summary
join_meta_hn[, .N, .(Name)] %>% .[N > 1] %>% merge(join_meta_hn) # -> not only SLF, but also AT&DE
join_meta_hs[, .N, .(Name)] %>% .[N > 1] %>% merge(join_meta_hs) # -> not only SLF, but also AT&DE

need_suffix_hn <- join_meta_hn[, .N, .(Name)] %>% .[N > 1, Name]
join_meta_hn[Name %in% need_suffix_hn, Name := paste0(Name, "_", provider)]
join_data_hn[variable %in% need_suffix_hn, variable := paste0(variable, "_", provider)]

need_suffix_hs <- join_meta_hs[, .N, .(Name)] %>% .[N > 1, Name]
join_meta_hs[Name %in% need_suffix_hs, Name := paste0(Name, "_", provider)]
join_data_hs[variable %in% need_suffix_hs, variable := paste0(variable, "_", provider)]

join_meta_hn[, .N, .(Name)] %>% summary
join_meta_hs[, .N, .(Name)] %>% summary

# stns meta == data
join_data_hn[variable == "Valloire_cl", variable := "Valloire_forest"]

# issue with "na" in some merged data
stns_data_hn <- unique(join_data_hn$variable)
missing_meta_hn <- sort(stns_data_hn[! stns_data_hn %in% join_meta_hn$Name])
missing_meta_hn_rename <- sort(join_meta_hn[provider %in% c("CH_METEOSWISS", "FR_METEOFRANCE") & 
                                              grepl("na", Name), 
                                            Name])
names(missing_meta_hn_rename) <- missing_meta_hn
join_data_hn[variable %in% missing_meta_hn, variable := missing_meta_hn_rename[variable]]

stns_data_hn <- unique(join_data_hn$variable)
join_meta_hn[Name %in% stns_data_hn]
length(stns_data_hn)
join_meta_hn <- join_meta_hn[Name %in% stns_data_hn]


join_data_hs[variable == "Rifugio_camuli", variable := "Rifugio_Nacamuli"]
stns_data_hs <- unique(join_data_hs$variable)
missing_meta_hs <- sort(stns_data_hs[! stns_data_hs %in% join_meta_hs$Name])
missing_meta_hs_rename <- sort(join_meta_hs[provider %in% c("CH_METEOSWISS", "FR_METEOFRANCE", 
                                                            "IT_TN", "IT_VDA_AIBM", "IT_VDA_CF") &
                                              grepl("na", Name), 
                                            Name])
names(missing_meta_hs_rename) <- missing_meta_hs
join_data_hs[variable %in% missing_meta_hs, variable := missing_meta_hs_rename[variable]]

stns_data_hs <- unique(join_data_hs$variable)
join_meta_hs[Name %in% stns_data_hs]
length(stns_data_hs)
join_meta_hs <- join_meta_hs[Name %in% stns_data_hs]


# remove years at start and end with all NA values
no_na_data_hn <- join_data_hn[!is.na(value)]
no_na_data_hn[, 
              .(Date = seq(make_date(min(year(Date)), 1, 1),
                           make_date(max(year(Date)), 12, 31),
                           by = "day")),
              .(provider, variable)] %>% 
  merge(no_na_data_hn, by = c("provider", "variable", "Date"), all = T) -> complete_data_hn

no_na_data_hs <- join_data_hs[!is.na(value)]
no_na_data_hs[, 
              .(Date = seq(make_date(min(year(Date)), 1, 1),
                           make_date(max(year(Date)), 12, 31),
                           by = "day")),
              .(provider, variable)] %>% 
  merge(no_na_data_hs, by = c("provider", "variable", "Date"), all = T) -> complete_data_hs


# only 1 value per date
check_nn_hn <- complete_data_hn[, .N, .(variable, Date)]
summary(check_nn_hn)
check_nn_hs <- complete_data_hs[, .N, .(variable, Date)]
summary(check_nn_hs)


# split and write ---------------------------------------------------------



# past <= 1960
past_hn <- complete_data_hn[year(Date) <= 1960]
past_hn_stns <- unique(past_hn$variable)
out_data_wide_hn <- dcast(past_hn, Date ~ variable, value.var = "value")
out_meta_wide_hn <- join_meta_hn[Name %in% past_hn_stns,
                                 .(Provider = provider, Name, Longitude, Latitude, Elevation)]

past_hs <- complete_data_hs[year(Date) <= 1960]
past_hs_stns <- unique(past_hs$variable)
out_data_wide_hs <- dcast(past_hs, Date ~ variable, value.var = "value")
out_meta_wide_hs <- join_meta_hs[Name %in% past_hs_stns,
                                 .(Provider = provider, Name, Longitude, Latitude, Elevation)]

out_data_long <- merge(past_hn[, .(Provider = provider, Name = variable, Date, HN = value)],
                       past_hs[, .(Provider = provider, Name = variable, Date, HS = value)],
                       by = c("Provider", "Name", "Date"), all = T)
setkey(out_data_long, Provider, Name, Date)
out_meta_long <- unique(rbind(out_meta_wide_hn, out_meta_wide_hs))
setkey(out_meta_long, Provider, Name)

path_out <- "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1787hn_1879hs-1960/r-data/"
saveRDS(out_data_long, file = path(path_out, "data_long_HN_HS.rds"))
saveRDS(out_data_wide_hn, file = path(path_out, "data_wide_HN.rds"))
saveRDS(out_data_wide_hs, file = path(path_out, "data_wide_HS.rds"))
saveRDS(out_meta_long, file = path(path_out, "meta_long_HN_HS.rds"))
saveRDS(out_meta_wide_hn, file = path(path_out, "meta_wide_HN.rds"))
saveRDS(out_meta_wide_hs, file = path(path_out, "meta_wide_HS.rds"))

path_out <- "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1787hn_1879hs-1960/csv/"
# fwrite(out_data_long, file = path(path_out, "data_long_HN_HS.csv")) # -> too big, and not needed
fwrite(out_data_wide_hn, file = path(path_out, "data_wide_HN.csv"))
fwrite(out_data_wide_hs, file = path(path_out, "data_wide_HS.csv"))
# fwrite(out_meta_long, file = path(path_out, "meta_long_HN_HS.csv")) # consistent with no data
fwrite(out_meta_wide_hn, file = path(path_out, "meta_wide_HN.csv"))
fwrite(out_meta_wide_hs, file = path(path_out, "meta_wide_HS.csv"))


# present >= 1961
present_hn <- complete_data_hn[year(Date) >= 1961]
present_hn_stns <- unique(present_hn$variable)
out_data_wide_hn <- dcast(present_hn, Date ~ variable, value.var = "value")
out_meta_wide_hn <- join_meta_hn[Name %in% present_hn_stns,
                                 .(Provider = provider, Name, Longitude, Latitude, Elevation)]

present_hs <- complete_data_hs[year(Date) >= 1961]
present_hs_stns <- unique(present_hs$variable)
out_data_wide_hs <- dcast(present_hs, Date ~ variable, value.var = "value")
out_meta_wide_hs <- join_meta_hs[Name %in% present_hs_stns,
                                 .(Provider = provider, Name, Longitude, Latitude, Elevation)]

out_data_long <- merge(present_hn[, .(Provider = provider, Name = variable, Date, HN = value)],
                       present_hs[, .(Provider = provider, Name = variable, Date, HS = value)],
                       by = c("Provider", "Name", "Date"), all = T)
setkey(out_data_long, Provider, Name, Date)
out_meta_long <- unique(rbind(out_meta_wide_hn, out_meta_wide_hs))
setkey(out_meta_long, Provider, Name)

path_out <- "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1961-2020/r-data/"
saveRDS(out_data_long, file = path(path_out, "data_long_HN_HS.rds"))
saveRDS(out_data_wide_hn, file = path(path_out, "data_wide_HN.rds"))
saveRDS(out_data_wide_hs, file = path(path_out, "data_wide_HS.rds"))
saveRDS(out_meta_long, file = path(path_out, "meta_long_HN_HS.rds"))
saveRDS(out_meta_wide_hn, file = path(path_out, "meta_wide_HN.rds"))
saveRDS(out_meta_wide_hs, file = path(path_out, "meta_wide_HS.rds"))

path_out <- "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1961-2020/csv/"
# fwrite(out_data_long, file = path(path_out, "data_long_HN_HS.csv")) # -> too big 1.4GB
fwrite(out_data_wide_hn, file = path(path_out, "data_wide_HN.csv"))
fwrite(out_data_wide_hs, file = path(path_out, "data_wide_HS.csv"))
# fwrite(out_meta_long, file = path(path_out, "meta_long_HN_HS.csv")) # consistent with no data
fwrite(out_meta_wide_hn, file = path(path_out, "meta_wide_HN.csv"))
fwrite(out_meta_wide_hs, file = path(path_out, "meta_wide_HS.csv"))




