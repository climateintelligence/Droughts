# plots of iv

library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(forcats)
library(foreach)
library(patchwork)
library(flextable)
library(officer)

dat_meta_clust <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/rds/meta-with-cluster-01.rds")
dat_month <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/rds/data-01-monthly.rds")
dat_seasonal <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/rds/data-02-seasonal.rds")

load("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/rds/trends-01-1971-2019-ols-gls.rda")

# rsq ---------------------------------------------------------------------

dat_month_gls[term == "year0", r.squared] %>% summary
dat_month_gls[term == "year0", r.squared] %>% qplot
dat_month_gls[term == "year0", mean(r.squared), month] 

dat_month_gls[term == "year0" & p.value < 0.05, r.squared] %>% summary
dat_month_gls[term == "year0" & p.value < 0.05, r.squared] %>% qplot
dat_month_gls[term == "year0" & p.value < 0.05, mean(r.squared), month] 


dat_month_ols[term == "year0"] %>% summary
dat_month_ols[term == "year0", r.squared] %>% qplot
dat_month_ols[term == "year0", mean(r.squared), month] 



dat_month_ols[term == "year0"] %>% 
  ggplot(aes(r.squared))+
  geom_histogram()+
  facet_wrap(~month)

dat_seasonal_ols[term == "year0", mean(r.squared), variable]

dat_seasonal_ols[term == "year0"] %>% 
  ggplot(aes(r.squared))+
  geom_histogram()+
  facet_wrap(~variable)

dat_month_gls[term == "year0" & p.value < 0.05] %>% 
  merge(dat_meta_clust) %>% 
  ggplot(aes(Elevation, r.squared))+
  geom_point()+
  facet_wrap(~month)

# residual error ----------------------------------------------------------

dat_plot1 <- dat_month_gls[term == "year0"] %>% 
  merge(dat_meta_clust) 
mitmatmisc::add_month_fct(dat_plot1, 10)
dat_plot1[, elev_fct := cut(Elevation, breaks = seq(0,3000,by=500), dig.lab=6)]

gg1 <-dat_plot1 %>% 
  ggplot(aes(49*estimate/resid.sd, elev_fct))+
  geom_vline(xintercept = 0, linetype = "dashed")+
  geom_boxplot(varwidth = T)+
  facet_wrap(~month_fct, nrow = 2)+
  theme_bw()+
  ylab("Elevation[m]")+
  xlab("Ratio of trend divided by sd of residuals")


dat_plot2 <- dat_seasonal_gls[term == "year0" & !startsWith(variable, "SCD")] %>% 
  merge(dat_meta_clust, by = "Name") 
dat_plot2[, elev_fct := cut(Elevation, breaks = seq(0,3000,by=500), dig.lab=6)]
dat_plot2[, variable_fct := fct_relevel(factor(variable), "maxHS_NDJFMAM", after = Inf)]

gg2 <- dat_plot2 %>% 
  ggplot(aes(49*estimate/resid.sd, elev_fct))+
  geom_vline(xintercept = 0, linetype = "dashed")+
  geom_boxplot(varwidth = T)+
  facet_grid(.~variable_fct)+
  theme_bw()+
  ylab("Elevation[m]")+
  xlab("Ratio of trend divided by sd of residuals")


dat_plot3 <- dat_seasonal_ols[term == "year0" & startsWith(variable, "SCD")] %>% 
  merge(dat_meta_clust, by = "Name") 
dat_plot3[, elev_fct := cut(Elevation, breaks = seq(0,3000,by=500), dig.lab=6)]
dat_plot3[, variable_fct := fct_relevel(factor(variable), "SCD_NDJF")]

gg3 <- dat_plot3 %>% 
  ggplot(aes(49*estimate/resid.sd, elev_fct))+
  geom_vline(xintercept = 0, linetype = "dashed")+
  geom_boxplot(varwidth = T)+
  facet_grid(.~variable_fct)+
  theme_bw()+
  ylab("Elevation[m]")+
  xlab("Ratio of trend divided by sd of residuals")


gg_out <- (gg1 / gg2 / gg3) + 
  plot_layout(heights = c(2,1,1)) + 
  plot_annotation(tag_levels = "a", tag_prefix = "(", tag_suffix = ")")


ggsave(gg_out,
       filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/fig/Figure B2.png",
       width = 10, height = 10)


# gamma (month) -------------------------------------------------------------------



dat_plot <- dat_month_gls[!is.na(cf.var) & term == "year0"] %>% 
  merge(dat_month[, .(meanHS = mean(HS)), .(Name, month)]) %>% 
  merge(dat_meta_clust)
mitmatmisc::add_month_fct(dat_plot, 10)

dat_plot[meanHS > 1] %>% 
  ggplot(aes(meanHS, cf.var, colour = cluster_fct))+
  geom_hline(yintercept = 0, linetype = "dashed")+
  geom_point(alpha = 0.6)+
  # geom_smooth(se = F)+
  scale_color_brewer("", palette = "Set1", guide = F)+
  facet_wrap(~month_fct, scales = "free_x", nrow = 2)+
  theme_bw()+
  xlab("meanHS [cm]")+
  ylab("Time coefficient of residual variance")

# ggsave(filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/fig/review-coauthors/gls-var-coef-meanHS.png",
#        width = 12, height = 8)

# 
# # summary table 1
# 
# 
# 
# dat_table <- dat_plot[meanHS > 1 & p.value.var < 0.05, 
#                       .(mean_gamma = mean(cf.var),
#                         nn = .N), 
#                       .(month_fct, cluster_fct)]
# 
# dat_table[, frac_var_change := exp(2*49*mean_gamma)]
# dat_table[, mean_gamma_sign := ifelse(sign(mean_gamma) >= 1, "+", "-")]
#   
# dat_table %>% 
#   dcast(cluster_fct ~ month_fct, value.var = "mean_gamma_sign") %>% 
#   flextable() %>% 
#   colformat_num(j = c(2:8)) %>% 
#   autofit()



# ** summary table --------------------------------------------------------

dat_table <- copy(dat_plot)
dat_table[, var_sign := ifelse(cf.var <= 0, "neg", "pos")]

dat_table_sig <- dat_table[, 
                           .(nn = .N), 
                           .(month_fct, cluster_fct, p_sig = p.value.var < 0.05, var_sign)]
dat_table_sig[, perc := nn / sum(nn), .(month_fct, cluster_fct)]

dat_table_sig[p_sig == T] %>% 
  dcast(cluster_fct ~ month_fct + var_sign, value.var = "perc") -> dat_table_out

dat_header <- data.table(col_keys = names(dat_table_out))
dat_header[, c("row1", "row2") := tstrsplit(col_keys, "_")]
dat_header[1, ":="(row1 = c("Region"), row2 = c(""))]
row2_rename <- setNames(c("#", "mean", "sig-", "sig+"),
                        c("nn", "mu", "neg", "pos"))
dat_header[, row2 := row2_rename[row2]]
cols_perc <- dat_header[row2 %in% c("sig-", "sig+"), col_keys]
dat_table_out[, c(cols_perc) := lapply(.SD, scales::percent, .1), .SDcols = cols_perc]

dat_table_out[, cluster_fct := fct_recode(cluster_fct,
                                          "N&hA" = "North & high Alpine",
                                          "S&hA" = "South & high Alpine")]

ft <-
  dat_table_out %>% 
  flextable() %>%
  set_header_df(dat_header) %>% 
  theme_booktabs() %>%
  vline(j = 1 + 0:6*2, border = fp_border()) %>% 
  merge_h(part = "header") %>%
  align(align = "left", part = "all") %>%
  # align(j = dat_header[1, col_keys], align = "left", part = "all") %>% 
  # align(j = dat_header[-c(1), col_keys], align = "center", part = "all") %>% 
  fix_border_issues() %>% 
  fontsize(size = 9, part = "all") %>% 
  font(fontname = "Times New Roman", part = "all") %>% autofit() 



read_docx() %>%
  body_add_flextable(ft) %>%
  print(target = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/table/Table B3.docx")


# ** numbers -----------------------------------------------------------------

dat_table[, 
          .(nn = .N), 
          keyby = .(month_fct, p_sig = p.value.var < 0.05, var_sign)] %>% 
  .[, perc := nn / sum(nn), .(month_fct)] %>% 
  .[]


# gamma (seasonalHS) -------------------------------------------------------------------

# not sure, if plot needed

dat_plot <- dat_seasonal_gls[!is.na(cf.var) & term == "year0" & !startsWith(variable, "SCD")] %>% 
  merge(dat_seasonal[, .(mean_value = mean(value)), .(Name, variable)]) %>% 
  merge(dat_meta_clust)
dat_plot[, variable_fct := fct_relevel(factor(variable), "maxHS_NDJFMAM", after = Inf)]

dat_plot[mean_value > 1] %>% 
  ggplot(aes(mean_value, cf.var, colour = cluster_fct))+
  geom_hline(yintercept = 0, linetype = "dashed")+
  geom_point(alpha = 0.6)+
  # geom_smooth(se = F)+
  scale_color_brewer("", palette = "Set1", guide = F)+
  facet_grid(variable_fct ~ cluster_fct, scales = "free_x")+
  theme_bw()+
  xlab("meanHS [cm]")+
  ylab("Time coefficient of residual variance")


dat_plot[mean_value > 0] %>% 
  ggplot(aes(Elevation, cf.var, colour = cluster_fct))+
  geom_hline(yintercept = 0, linetype = "dashed")+
  geom_point(alpha = 0.6)+
  # geom_smooth(se = F)+
  scale_color_brewer("", palette = "Set1", guide = F)+
  facet_wrap(~variable_fct, scales = "free_x")+
  theme_bw()+
  xlab("meanHS [cm]")+
  ylab("Time coefficient of residual variance")

# ggsave(filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/fig/review-coauthors/gls-var-coef-meanHS.png",
#        width = 12, height = 8)



# ** summary table --------------------------------------------------------

dat_table <- copy(dat_plot)
dat_table[, var_sign := ifelse(cf.var <= 0, "neg", "pos")]

dat_table_sig <- dat_table[, 
                           .(nn = .N), 
                           .(variable_fct, cluster_fct, p_sig = p.value.var < 0.05, var_sign)]
dat_table_sig[, perc := nn / sum(nn), .(variable_fct, cluster_fct)]

dat_table_sig[p_sig == T] %>% 
  dcast(cluster_fct ~ variable_fct + var_sign, value.var = "perc") -> dat_table_out

dat_header <- data.table(col_keys = names(dat_table_out))
dat_header[, row1 := stringr::str_sub(col_keys, 0, -5)]
dat_header[, row2 := stringr::str_sub(col_keys, -3, -1)]

dat_header[1, ":="(row1 = c("Region"), row2 = c(""))]
row2_rename <- setNames(c("#", "mean", "sig-", "sig+"),
                        c("nn", "mu", "neg", "pos"))
dat_header[, row2 := row2_rename[row2]]
cols_perc <- dat_header[row2 %in% c("sig-", "sig+"), col_keys]
dat_table_out[, c(cols_perc) := lapply(.SD, scales::percent, .1), .SDcols = cols_perc]

dat_table_out[, cluster_fct := fct_recode(cluster_fct,
                                          "N&hA" = "North & high Alpine",
                                          "S&hA" = "South & high Alpine")]

ft <-
  dat_table_out %>% 
  flextable() %>%
  set_header_df(dat_header) %>% 
  theme_booktabs() %>%
  vline(j = 1 + 0:4*2, border = fp_border()) %>% 
  merge_h(part = "header") %>%
  align(align = "left", part = "all") %>%
  # align(j = dat_header[1, col_keys], align = "left", part = "all") %>% 
  # align(j = dat_header[-c(1), col_keys], align = "center", part = "all") %>% 
  fix_border_issues() %>% 
  fontsize(size = 9, part = "all") %>% 
  font(fontname = "Times New Roman", part = "all") %>% autofit() 


# 
# read_docx() %>% 
#   body_add_flextable(ft) %>% 
#   print(target = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/table/Table 2.docx")






