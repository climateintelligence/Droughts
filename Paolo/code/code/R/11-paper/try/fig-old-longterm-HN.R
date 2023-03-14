# prep figure on longterm HS/HN changes

library(data.table)
library(magrittr)
library(ggplot2)
library(forcats)
library(slider)


load("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/data-longterm-HN-HS.rda")


# plot HN ------------------------------------------------------------

dat_hn %>% 
  merge(dat_meta_hn) %>% 
  
  ggplot(aes(hydro_year, HN))+
  geom_line(aes(group = Name), colour = "grey30")+
  theme_bw()+
  theme(panel.grid.minor = element_blank())+
  facet_wrap(~ fct_reorder(Name, HN, max, na.rm = T), scales = "free_y")+
  ylim(0, NA)+
  xlab(NULL)+
  ylab("Seasonal sum of fresh snow [cm]")

ggsave(filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/fig/longterm-ts-HN.png",
       width = 12, height = 6)

# 
# dat_hs %>% 
#   merge(dat_meta_hs) %>% 
#   
#   ggplot(aes(hydro_year, HS))+
#   geom_line(aes(group = Name), colour = "grey30")+
#   theme_bw()+
#   theme(panel.grid.minor = element_blank())+
#   facet_wrap(~ fct_reorder(Name, HS, max, na.rm = T), scales = "free_y")+
#   ylim(0, NA)+
#   xlab(NULL)+
#   ylab("Seasonal mean snow depth [cm]")


