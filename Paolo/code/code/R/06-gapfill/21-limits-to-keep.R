# identify limis on what gapfill to keep


library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(fs)


# data --------------------------------------------------------------------

dat_meta <-  readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/rds/1961-2020/meta_wide_HS.rds")

dat <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/04_GAPFILL/rds/gapfill-01-all.rds")

  
dat_n_filled <- dat[,
                    .(mean_HS = mean(HS, na.rm = T),
                      mean_HS_noNA = mean(HS),
                      mean_HS_original = mean(HS[HS_fillcode == 1], na.rm = T),
                      n_days = .N,
                      n_avail_total = sum(!is.na(HS)),
                      n_miss_total = sum(is.na(HS)),
                      n_avail_original = sum(HS_fillcode == 1, na.rm = T),
                      n_filled = sum(HS_fillcode == 222, na.rm = T)),
                    .(Name, year(Date), month(Date))]




# plot stn overview (~1:30h) -------------------------------------------------------

dat_n_filled[n_filled > 0, .(Name)] %>% 
  unique -> stns_filled
# -> almost all anyway

for(i_prov in sort(unique(dat_meta$Provider))){
  
  pdf(path("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/04_GAPFILL/fig/stn-overview-fill/",
           i_prov, ext = "pdf"),
      width = 12, height = 6)
  
  for(i_stn in dat_meta[Provider == i_prov, Name]){
    
    dat_n_filled[Name == i_stn] %>% 
      melt(measure.vars = c("n_avail_original", "n_filled", "n_miss_total")) -> dat_plot
    mitmatmisc::add_month_fct(dat_plot, 10)
      
    gg <- dat_plot %>% 
      ggplot(aes(year, value, fill = variable))+
      geom_col(width = 1, linetype = "blank")+
      facet_wrap(~month_fct)+
      theme_bw()+
      scale_fill_brewer(palette = "Accent")+
      xlab(NULL)+
      ylab("# days")+
      ggtitle(i_stn)
    
    print(gg)
    
    
  }
  
  dev.off()
  
}





# look at influence of NA on mean -----------------------------------------

dat_n_filled[month == 1 & n_filled > 0 & n_miss_total == 0, .N, keyby = n_filled]

dat_n_filled[month == 1 & n_filled > 0 & n_miss_total == 0, 
             .(mean(mean_HS - mean_HS_original),
               mean((mean_HS_original - mean_HS) / mean_HS, na.rm = T)), 
             keyby = n_filled]

dat_n_filled[month == 1 & n_filled > 0 & n_miss_total == 0] %>% 
  ggplot(aes(as.factor(n_filled), (mean_HS_original - mean_HS)))+
  geom_boxplot()


dat_n_filled[n_filled > 0 & n_miss_total == 0 & n_filled <= 20, 
             .(delta = mean(mean_HS - mean_HS_original),
               delta_rel = mean((mean_HS_original - mean_HS) / mean_HS, na.rm = T)), 
             .(month, n_filled)] %>% 
  ggplot(aes(n_filled, delta))+
  geom_point()+
  facet_wrap(~month)




# some numbers ------------------------------------------

dat_n_filled[, 
             .(n_filled = sum(n_filled)),
             .(Name, month)] %>% 
  merge(dat_meta, by = "Name") %>% 
  .[, country := substr(Provider, 1, 2)] %>% 
  ggplot(aes(n_filled))+
  geom_histogram()+
  facet_grid(country ~ month, scales = "free_y")




dat_n_summary <- dat_n_filled[,
                              .(n_filled = sum(n_filled),
                                n_avail_original = sum(n_avail_original),
                                n_avail_total = sum(n_avail_total)),
                              .(Name, month)]
dat_n_summary[, frac_filled_total := n_filled / n_avail_total]
dat_n_summary[, frac_filled_original := n_filled / n_avail_original]
dat_n_summary %>% summary
mitmatmisc::add_month_fct(dat_n_summary, 10)

dat_n_summary %>% 
  ggplot(aes(month_fct, frac_filled_total))+
  geom_boxplot()

dat_n_summary[frac_filled_original > 100]
dat_n_summary[n_avail_original > 100] %>% summary

dat_n_summary[n_avail_original > 100] %>% 
  ggplot(aes(frac_filled_original))+
  geom_histogram()+
  facet_wrap(~month_fct)


dat_n_summary[n_avail_original > 100] %>% 
  ggplot(aes(month_fct, frac_filled_original))+
  geom_boxplot()



# probably need some thresholds on # consecutive gaps ---------------------

# -> also some years reconstruction? +- 5 years




