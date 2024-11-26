
install.packages("tidyverse")
library("tidyverse") 


mpg1 <- as.data.frame(mpg)
head(mpg1)
cars1 <- as_tibble(cars)
cars1

mtcars1 <- as_tibble(rownames_to_column(mtcars))
head(mtcars1)

filter(mtcars1, mpg<25, cyl %in% c(4,6), am==0)
filter(mtcars1, mpg >= quantile(mpg,0,25) & mpg <= quantile(mpg,0,75))

select(mtcars1, rowname, cyl:wt)

select(mtcars1, -rowname)

groupmtcars1 <- group_by(mtcars1, am, cyl)
groupmtcars1

summarise(mtcars1, n=n(), mean(mpg))
summarise(groupmtcars1, n=n(), mean(mpg))
summarise(groupmtcars3, n=n(), mean(mpg))

mtcars1 <- as_tibble(rownames_to_column(mtcars))
mtcars1
ggplot(mtcars1, aes(hp, mpg, color=factor(cyl))) + geom_point()

ggplot(mtcars1, aes(hp, mpg, color=factor(cyl))) + geom_smooth(aes(group=123), se=FALSE)

ggplot(mtcars1, aes(hp, mpg, color=factor(cyl))) + geom_point() + geom_smooth(aes(group=123), se=FALSE)

ggplot(mtcars1, aes(hp, mpg, color=(cyl))) + geom_point() +
labs(title="Relationship between Fuel efficiency and Horse", x= "horse power", y="mile per", caption = "data source", color="Cylinder") +
geom_label(aes(label=rowname), nudge_x = 2.5, nudge_y = 1.5, label.size = 0.24, show.legend = FALSE)

hmpg <- mtcars1 %>% group_by(cyl) %>% filter(row_number(desc(mpg))==1)


ggplot(mtcars1, aes(hp, mpg, color=factor(cyl))) + geom_point() +
labs(title="Relationship between Fuel efficiency and Horse", x= "horse power", y="mile per", caption = "data source", color="Cylinder") +
geom_label(aes(label=rowname), data=hmpg, nudge_x = 2.5, nudge_y = 1.5, label.size = 0.24, show.legend = FALSE)

ggplot(mtcars1, aes(hp, mpg, color=factor(cyl))) + geom_point() +
labs(title="Relationship between Fuel efficiency and Horse", x= "horse power", y="mile per", caption = "data source", color="Cylinder") +
scale_x_continuous(labels=NULL) + scale_y_continuous(labels= NULL) + scale_color_discrete(labels= NULL)

ggplot(mtcars1, aes(hp, mpg, color=factor(cyl))) + geom_point() +
labs(title="Relationship between Fuel efficiency and Horse", x= "horse power", y="mile per", caption = "data source", color="Cylinder") +
scale_x_continuous(labels=NULL) + scale_y_continuous(labels=NULL) 

ggplot(mtcars1, aes(hp, mpg, color=factor(cyl))) + geom_point() +
labs(title = "Relationship between Fuel efficiency and Horse", x="horse power", y="mile per", color="Cylinder") +
scale_y_continuous(breaks = seq(10, 40, by=10))

x_axis <- scale_x_continuous(limits = range(mtcars1$hp))
y_axis <- scale_y_continuous(limits = range(mtcars1$mpg))
col_legend <- scale_colour_discrete(limits = unique(factor(mtcars1$cyl)))

ggplot(mtcars1, aes(hp, mpg, color=factor(cyl))) +
geom_point() + geom_smooth(aes(group=123), se=FALSE) +
labs(title = "Relationship between Fuel efficiency and Horse", x="horse power", y="mile per", color="Cylinder") +
coord_cartesian(xlim=c(100,250), ylim = c(10,30))


ggplot(mtcars1, aes(hp, mpg, color=factor(cyl))) +
geom_point() +
labs(title = "Relationship between Fuel efficiency and Horse", x="horse power", y="mile per", color="Cylinder") +
theme(legend.position = "top")


ggplot(mtcars1, aes(hp, mpg, color=factor(cyl))) +
geom_point() +
labs(title = "Relationship between Fuel efficiency and Horse", x="horse power", y="mile per", color="Cylinder") +
guides(color = guide_legend(nrow=3, override.aes = list(size = 5))) +
theme_classic() +
theme(legend.position = "bottom")



