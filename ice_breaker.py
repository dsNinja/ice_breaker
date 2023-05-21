from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv

from third_parties.linkedin import scrap_linkedin_profile

load_dotenv()

information = """
    Michael Jeffrey Jordan (born February 17, 1963), also known by his initials MJ,[9] is an American former professional basketball player and businessman. His biography on the official National Basketball Association (NBA) website states: "By acclamation, Michael Jordan is the greatest basketball player of all time."[10] He played fifteen seasons in the NBA, winning six NBA championships with the Chicago Bulls. Jordan is the principal owner and chairman of the Charlotte Hornets of the NBA and of 23XI Racing in the NASCAR Cup Series. He was integral in popularizing the basketball sport and the NBA around the world in the 1980s and 1990s,[11] becoming a global cultural icon.[12]

    Jordan played college basketball for three seasons under coach Dean Smith with the North Carolina Tar Heels. As a freshman, he was a member of the Tar Heels' national championship team in 1982.[5] Jordan joined the Bulls in 1984 as the third overall draft pick[5][13] and quickly emerged as a league star, entertaining crowds with his prolific scoring while gaining a reputation as one of the game's best defensive players.[14] His leaping ability, demonstrated by performing slam dunks from the free-throw line in Slam Dunk Contests, earned him the nicknames "Air Jordan" and "His Airness".[5][13] Jordan won his first NBA title with the Bulls in 1991 and followed that achievement with titles in 1992 and 1993, securing a three-peat. Jordan abruptly retired from basketball before the 1993–94 NBA season to play Minor League Baseball but returned to the Bulls in March 1995 and led them to three more championships in 1996, 1997, and 1998, as well as a then-record 72 regular season wins in the 1995–96 NBA season.[5] He retired for the second time in January 1999 but returned for two more NBA seasons from 2001 to 2003 as a member of the Washington Wizards.[5][13] During the course of his professional career, he was also selected to play for the United States national team, winning four gold medals—at the 1983 Pan American Games, 1984 Summer Olympics, 1992 Tournament of the Americas and 1992 Summer Olympics—while also being undefeated.[15]

    Jordan's individual accolades and accomplishments include six NBA Finals Most Valuable Player (MVP) awards, ten NBA scoring titles (both all-time records), five NBA MVP awards, ten All-NBA First Team designations, nine All-Defensive First Team honors, fourteen NBA All-Star Game selections, three NBA All-Star Game MVP awards, three NBA steals titles, and the 1988 NBA Defensive Player of the Year Award.[13] He holds the NBA records for career regular season scoring average (30.12 points per game) and career playoff scoring average (33.4 points per game).[16] In 1999, he was named the 20th century's greatest North American athlete by ESPN and was second to Babe Ruth on the Associated Press' list of athletes of the century.[5] Jordan was twice inducted into the Naismith Memorial Basketball Hall of Fame, once in 2009 for his individual career,[17] and again in 2010 as part of the 1992 United States men's Olympic basketball team ("The Dream Team").[18] He became a member of the United States Olympic Hall of Fame in 2009,[19] a member of the North Carolina Sports Hall of Fame in 2010,[20] and an individual member of the FIBA Hall of Fame in 2015 and a "Dream Team" member in 2017.[21][22] In 2021, he was named to the NBA 75th Anniversary Team.[23]

    One of the most effectively marketed athletes of his generation, Jordan is known for his product endorsements.[11][24] He fueled the success of Nike's Air Jordan sneakers, which were introduced in 1984 and remain popular today.[25] He starred as himself in the live-action/animation hybrid film Space Jam (1996) and was the central focus of the Emmy-winning documentary series The Last Dance (2020). He became part-owner and head of basketball operations for the Charlotte Bobcats (now named the Hornets) in 2006 and bought a controlling interest in 2010. In 2016, he became the first billionaire player in NBA history.[26] That year, President Barack Obama awarded him the Presidential Medal of Freedom.[27] As of 2023, his net worth is estimated at $2 billion.[28]
"""


if __name__ == "__main__":
    print("Hello LangChain!")

    summary_template = """
        given the LinkedIn information {information} about a person from I want you to create:
        1. short summary
        2. experience in the format of key value pairs of role and years
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    data = scrap_linkedin_profile()

    print(chain.run(information=data))
