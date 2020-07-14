#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

setwd("C:\\Users\\HP\\Documents\\Courses\\digitalvidya\\datascienceusingr\\class\\week6\\data")
#getwd()
library(shiny)
library(shinydashboard)
library(dashboard)
library(plotly)
library(ggplot2)
library(readxl)
library(dplyr)
library(DT)
#retrieving the data
bat_scored <- read_excel("Batsman_Scored.xlsx")
ball_by_ball <- read_excel("Ball_by_Ball.xlsx")
player_matches <- read_excel("Player_Match.xlsx")
players <- read_excel("Player.xlsx")
matches <- read_excel("Match.xlsx")
seasons <- read_excel("Season.xlsx")
wickets <- read_excel("Wicket_Taken.xlsx")
teams <- read_excel("Team.xlsx")

#merging the data together
mer1 <- merge(bat_scored,ball_by_ball)
mer2 <- merge(mer1,players,by.x = "striker",by.y = "player_id")
mer3 <- merge(mer2,matchs)
mer4 <- merge(mer3,seasons)
mer5 <- merge(mer4,wickets)

ui <- dashboardPage(
  dashboardHeader(title="IPL DASHBOARD"),
  dashboardSidebar(
    sliderInput(inputId = "TopN",label="Top N Batsmen", min=1, max = 50 ,value = 5),
    sidebarMenu(
      menuItem(tabName = "q1","Question a"),
      menuItem(tabName = "q2","Question b"),
      menuItem(tabName = "q3","Question c")
    )),
  dashboardBody(theme_purple_gradient,
                tabItems(
                  tabItem(tabName = "q1",
                          fluidRow(
                            box(title="Top N by runs",width=6,collapsible=T,plotlyOutput("plot1")),
                            box(title="Top N by batting average",width=6,collapsible=T,plotlyOutput("plot2"))),
                          fluidRow(
                            box(title="Top N by Strike Rate",width=6,collapsible=T,plotlyOutput("plot3")),
                            box(title="Top N by Highest Score",width=6,collapsible=T,plotlyOutput("plot4")))),
                  tabItem(tabName = "q2",
                          selectInput(inputId = "Year",label = "Select the Year",choices = c("2008","2009","2010","2011","2012","2013","2014","2015","2016"),selected = "2008"),
                          fluidRow(
                            valueBoxOutput("value1"),
                            valueBoxOutput("value2"),
                            valueBoxOutput("value3")),
                          fluidRow(
                            valueBoxOutput("value4"),
                            valueBoxOutput("value5"),
                            valueBoxOutput("value6"))),
                  tabItem(tabName = "q3",
                          fluidRow(
                            box(plotlyOutput("plot5")),
                            box(dataTableOutput("table2")))))
  ))

server <- function(input, output){
  
  output$plot1 <- renderPlotly({
    p <- mer2 %>% group_by(player_name) %>% summarise(runs=sum(runs_scored)) %>% arrange(-runs) %>% head(input$TopN) %>% ggplot(aes(x=reorder(player_name,-runs),y=runs,fill=player_name))+geom_bar(stat = "identity")+theme(axis.text.y = element_blank(),axis.title.y = element_blank(),axis.text.x = element_blank())+labs(x="Player Name")
    ggplotly(p)
  })
  output$plot2 <- renderPlotly({
    p <- mer2 %>% group_by(player_name) %>% summarise(avg=sum(runs_scored)/length(unique(match_id)))%>% arrange(-avg) %>% head(input$TopN) %>% ggplot(aes(x=reorder(player_name,-avg),y=avg,fill=player_name))+geom_bar(stat = "identity")+theme(axis.text.y = element_blank(),axis.title.y = element_blank(),axis.text.x = element_blank())+labs(x="Player Name")
    ggplotly(p)
  })
  output$plot3 <- renderPlotly({
    balls_faced <- mer2 %>% group_by(player_name) %>% summarise(balls=n())%>% filter(balls>=500)
    strike_rate <- mer2 %>% filter(player_name %in% balls_faced$player_name) %>%  group_by(player_name) %>% summarise(strike=mean(runs_scored)*100) %>% arrange(-strike) %>% head(input$TopN)
    p <-ggplot(strike_rate,aes(x=reorder(player_name,-strike),y=strike,fill=player_name))+geom_bar(stat = "identity")+theme(axis.text.y = element_blank(),axis.title.y = element_blank(),axis.text.x = element_blank())+labs(x="Player Name")
    ggplotly(p)
  })
  output$plot4 <- renderPlotly({
    p <- mer2 %>% group_by(player_name,match_id) %>% summarise(score=sum(runs_scored)) %>% arrange(-score) %>% head(input$TopN) %>% ggplot(aes(x=reorder(player_name,-score),y=score,fill=player_name))+geom_bar(stat = "identity")+theme(axis.text.y = element_blank(),axis.title.y = element_blank(),axis.text.x = element_blank())+labs(x="Player Name")
    ggplotly(p)
  })
  output$value1 <- renderValueBox({
    high_runs <- mer4 %>% filter(season_year==input$Year) %>% group_by(player_name) %>% summarise(h_runs=sum(runs_scored)) %>% arrange(-h_runs)
    valueBox(value =mer4[mer4$striker==mer4$orange_cap & mer4$season_year==input$Year,"player_name"][1],subtitle = paste("Number of runs scored:",high_runs[1,2]),color = "orange")
  })
  output$value2 <- renderValueBox({
    bow_wkt <- mer5 %>% filter(season_year==input$Year) %>% group_by(bowler) %>% summarise(wkts=length(player_out)) %>% arrange(-wkts)
    valueBox(value = mer4[mer4$striker==mer4$purple_cap & mer4$season_year==input$Year,"player_name"][1],subtitle = paste("Number of Wickets taken:",bow_wkt[1,2]),color = "purple")
  })
  output$value3 <- renderValueBox({
    valueBox(value = mer4 %>% filter(season_year==input$Year) %>% summarise(total=length(unique(match_id))),subtitle = "Total Number of matches played")
  })
  output$value4 <- renderValueBox({
    valueBox(value = mer4 %>% filter(season_year==input$Year) %>% summarise(sixes=sum(runs_scored==6)),subtitle = "Total Number of 6s")
  })
  output$value5 <- renderValueBox({
    valueBox(value = mer4 %>% filter(season_year==input$Year) %>% summarise(fours=sum(runs_scored==4)),subtitle = "Total Number of 4s")
  })
  output$value6 <- renderValueBox({
    valueBox(value = mer5 %>% filter(season_year==input$Year) %>% summarise(wkt=length(player_out)),subtitle = "Total Number of wickets")
  })
  output$plot5 <- renderPlotly({
    j=1
    for(i in matches$match_winner){
      matches$match_winner_name[j] <- teams$team_name[team$team_id==i]
      j=j+1
    }
    
    
    j=1
    team_name=c()
    for(i in unique(matches$season_id)){
      mer6 <- matches %>% filter(season_id==i)
      team_name[j] <- mer6$match_winner_name[nrow(m6)]
      j=j+1
    }
    title <- data.frame(team_name)
    title1 <- title %>% group_by(team_name) %>% summarise(count=n()) %>% arrange(-count)
    
    
    p <- plot_ly(title1, labels = ~team_name, values = ~count, type = 'pie') %>%
      layout(title = 'Team wise title count',
             xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
             yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE))
    p
  })
  output$table2 <- renderDataTable({
    j=1
    for(i in matches$match_winner){
      matches$match_winner_name[j] <- teams$team_name[team$team_id==i]
      j=j+1
    }
    
    
    j=1
    team_name=c()
    for(i in unique(match$season_id)){
      mer7 <- matches %>% filter(season_id==i)
      team_name[j] <- mer7$match_winner_name[nrow(m6)]
      j=j+1
    }
    title <- data.frame(team_name)
    title1 <- title %>% group_by(team_name) %>% summarise(count=n()) %>% arrange(-count)
    title1
  })
}

shinyApp(ui,server)
