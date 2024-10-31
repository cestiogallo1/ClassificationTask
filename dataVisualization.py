import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("./Dataset/Airlines.csv")

    # Correlazione Delay e Airline
    dt_df = df[['Airline', 'Delay']].groupby('Airline').agg('count').reset_index()
    plt.bar(dt_df["Airline"],dt_df['Delay'])
    plt.xlabel("Airline")
    plt.ylabel("Delay")
    plt.savefig("./Immagini/AirlineDelayCorrelation.png")
    plt.clf()
    
    # Correlazione Delay e AirportFrom
    dt_df = df[['AirportFrom', 'Delay']].groupby('AirportFrom').agg('count').reset_index()
    plt.bar(dt_df["AirportFrom"],dt_df['Delay'])
    plt.xlabel("AirportFrom")
    plt.ylabel("Delay")
    plt.savefig("./Immagini/AirportFromDelayCorrelation.png")
    plt.clf()
    
    # Correlazione Delay e AirportTo
    dt_df = df[['AirportTo', 'Delay']].groupby('AirportTo').agg('count').reset_index()
    plt.bar(dt_df["AirportTo"],dt_df['Delay'])
    plt.xlabel("AirportTo")
    plt.ylabel("Delay")
    plt.savefig("./Immagini/AirportToDelayCorrelation.png")
    plt.clf()
    
    # Correlazione Delay e Time
    dt_df = df[['Time', 'Delay']].groupby('Time').agg('count').reset_index()
    plt.bar(dt_df["Time"],dt_df['Delay'])
    plt.xlabel("Time")
    plt.ylabel("Delay")
    plt.savefig("./Immagini/TimeDelayCorrelation.png")
    plt.clf()


    # Correlazione Delay e Giorno Settimana
    dow_df = df[['DayOfWeek', 'Delay']].groupby('DayOfWeek').agg('count').reset_index()
    plt.bar(dow_df["DayOfWeek"],dow_df['Delay'])
    plt.xlabel("DayOfWeek")
    plt.ylabel("Delay")
    plt.savefig("./Immagini/DayOfWeekDelayCorrelation.png")
    plt.clf()

    # Correlazione Delay e Distanza
    dist_df = df[['Length', 'Delay']].groupby('Length').agg('count').reset_index()
    plt.bar(dist_df["Length"],dist_df['Delay'])
    plt.xlabel("Length")
    plt.ylabel("Delay")
    plt.savefig("./Immagini/DistanzaDelayCorrelation.png")
    plt.clf()

    # Frequenza AirLine
    df.groupby('Airline').size().plot(kind='pie', autopct='%.2f', label='Airlines')
    plt.savefig("./Immagini/PieChartAirline.png")
    plt.clf()

    # PieChart Delay
    df.groupby('Delay').size().plot(kind='pie', autopct='%.2f', label='Delays')
    plt.savefig("./Immagini/PieChartDelay.png")





if __name__ == '__main__':
    main()
