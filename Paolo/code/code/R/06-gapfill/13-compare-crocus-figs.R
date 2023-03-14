# compare Crocus gapfill to Alice's functions



library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(patchwork)
library(scico)
library(forcats)
library(fs)
library(officer)
library(flextable)



# prep data ---------------------------------------------------------------

dat_meta <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/04_GAPFILL/rds/crocus-01-meta.rds")

dat_crocus <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/04_GAPFILL/rds/crocus-02-data.rds")
dat_gapfill <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/04_GAPFILL/rds/gapfill-01-all.rds")
dat_gapfill

stn_fr <- unique(dat_crocus$stn_name)
dat_gapfill2 <- dat_gapfill[Name %in% stn_fr]

dat_fr <- merge(dat_gapfill2[, .(stn_name = Name, date = Date, hs_fill = HS, hs_fillcode = HS_fillcode)],
                dat_crocus,
                by = c("stn_name", "date"),
                all = F)

dat_fr
mitmatmisc::add_hydro_year(dat_fr)


# plot common fill (>30min) ---------------------------------------------------------------


dat_fr[hs_fillcode == 222]
dat_fr[hs_fillcode == 222, .(stn_name, hydro_year)] %>% 
  unique -> dat_to_plot

cols <- setNames(c("black", scales::brewer_pal(palette = "Set1")(2)), 
                 c("hs_obs", "hs_fill", "hs_crocus_assim"))

for(
  i_stn_name in unique(dat_to_plot$stn_name)
){

  dat_i_stn <- dat_fr[stn_name == i_stn_name]
  dat_i_stn[hs_fillcode != 222, hs_fill := NA]
  
  ylim <- dat_i_stn[, range(c(hs_obs, hs_fill, hs_crocus_assim), na.rm = T)]
  out_file <- path("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/04_GAPFILL/fig/crocus-comparison/",
                    i_stn_name,
                    ext = "pdf")
  pdf(out_file, width = 12, height = 6)
  
  for(
    i_hydro_year in dat_to_plot[stn_name == i_stn_name, hydro_year]
  ){
    
    dat_plot <- dat_i_stn[hydro_year == i_hydro_year]
    gg <- dat_plot %>% 
      melt(measure.vars = c("hs_obs", "hs_fill", "hs_crocus_assim")) %>% 
      ggplot(aes(date, value, colour = variable, shape = variable))+
      geom_point()+
      scale_color_manual(values = cols)+
      scale_shape_manual(values = c(1, 3, 4))+
      ylim(ylim)+
      theme_bw()+
      xlab(NULL)+ylab("HS [cm]")
    
    print(gg)
    
  }   
  
  dev.off()

}


