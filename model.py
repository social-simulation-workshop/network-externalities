import argparse
import csv
import os
import numpy as np


class ArgsModel(object):
    
    def __init__(self) -> None:
        super().__init__()
    
        self.parser = argparse.ArgumentParser()
        self.parser = self.add_agent_param(self.parser)
        self.parser = self.add_internet_param(self.parser)
        self.parser = self.add_network_param(self.parser)
        self.parser = self.add_exp_param(self.parser)
    

    @staticmethod
    def add_agent_param(parser):
        parser.add_argument('--k', type=float, default=0.1,
            help="multiplicative constants for the pure income effect")
        parser.add_argument('--delta', type=float, default=0.1,
            help="multiplicative constants for network effect")
        parser.add_argument('--gamma', type=float, default=0.5,
            help="exponents of income")
        parser.add_argument('--alpha', type=float, default=0.5,
            help="exponents of the proportion of adopters")
        return parser
    

    @staticmethod
    def add_internet_param(parser):
        parser.add_argument('--p_0', type=float, default=60.0,
            help="the initial Internet price")
        parser.add_argument('--p_min', type=float, default=28.74,
            help="the equilibrium price level")
        parser.add_argument('--a', type=float, default=3.34,
            help="the speed of reversion to that equilibrium price")
        return parser
    

    @staticmethod
    def add_network_param(parser):
        parser.add_argument('--w_race', type=float, default=0.83,
            help="the weight of race")
        parser.add_argument('--w_edu', type=float, default=0.53,
            help="the weight of education")
        parser.add_argument('--w_inc', type=float, default=0.53,
            help="the weight of income")
        parser.add_argument('--h', type=float, default=0.,
            help="homophily bias")
        parser.add_argument('--is_spec_net', type=bool, default=True,
            help="is identity-specific net")
        parser.add_argument('--scaler', type=int, default=3,
            help="""the scaler to times the target number of relations for the agent i
                    as the size of the agent i's in-group.""")
        return parser
    

    @staticmethod
    def add_exp_param(parser):
        parser.add_argument('--n_period', type=int, default=100,
            help="the # of the period to simulate.")
        parser.add_argument('--seed', type=int, default=1,
            help="random seed.")
        parser.add_argument('--expNo', type=int, default=1,
            help="the number of the experiments to model one of the 7 scenarios.")
        return parser


    def get_args(self):
        return self.parser.parse_args()


class AgentDataHolder(object):
    
    def __init__(self, path_to_agentInfo) -> None:
        super().__init__()
        self.agents_data = self.pre_processing(path_to_agentInfo)
        self.race, self.edu, self.inc = self.getID_group_by_race(path_to_agentInfo)
    

    @staticmethod
    def pre_processing(path_to_info):
        """
        Return
        - agent_row -> 2d np.array, size=(# of agents, 6):
            [np.array([id, network_size, race, education, income, income_unstandardized]),
             np.array([id, network_size, race, education, income, income_unstandardized]),
             ....]
        """
        info_f = open(path_to_info, newline='')
        agent_rows = csv.reader(info_f)
        agents_data = None
        for row_idx, agent_row in enumerate(agent_rows):
            if row_idx == 0:
                col_names = agent_row
                continue
            else:
                agent_data = {col_n:col_val for col_n, col_val in zip(col_names, agent_row)}
        
            data = list()
            # id
            data.append(int(agent_data["Respondent id number"]))
            
            # network_size
            net_str = agent_data["How many friends close to discuss problems"]
            if net_str == "96 or higher":
                net_str = 96
            data.append(int(net_str))
            
            # race
            if agent_data["Race of respondent"] == "White":
                race = 1
            elif agent_data["Race of respondent"] == "Black":
                race = 0
            data.append(float(race))
            
            # education
            data.append(float(agent_data["Highest year of school completed"]))

            # income
            inc_str = agent_data["Total family income"]
            if inc_str == "Under 1000":
                inc_lower_bound, inc_upper_bound = 0, 999
            elif inc_str == "110000 or over":
                inc_lower_bound, inc_upper_bound = 110000, 650000
            else:
                inc_lower_bound, inc_upper_bound = inc_str.split("to")
                inc_lower_bound, inc_upper_bound = int(inc_lower_bound), int(inc_upper_bound)
            inc = np.random.randint(inc_lower_bound, inc_upper_bound)
            data.append(float(inc))

            # data
            data = np.array([data])
            agents_data = np.concatenate((agents_data, data), axis=0) if agents_data is not None else data
        
        # standardize race, education, income
        inc_unstandard = np.copy(agents_data[:, 4]).reshape((-1, 1))
        for idx in range(2, 5):
            fea_arr = np.copy(agents_data[:, idx])
            agents_data[:, idx] = (fea_arr-np.mean(fea_arr))/np.std(fea_arr)
        agents_data = np.concatenate((agents_data, inc_unstandard), axis=1)

        print("agents_data size: {}".format(agents_data.shape))
        return agents_data


    @staticmethod
    def getID_group_by_race(path_to_info):
        white, black = list(), list()
        college, high_school = list(), list()
        high, low = list(), list()

        info_f = open(path_to_info, newline='')
        agent_rows = csv.reader(info_f)
        for row_idx, agent_row in enumerate(agent_rows):
            if row_idx == 0:
                col_names = agent_row
                continue
            else:
                agent_data = {col_n:col_val for col_n, col_val in zip(col_names, agent_row)}

            agent_id = int(agent_data["Respondent id number"])
            
            # race: black v.s. white
            if agent_data["Race of respondent"] == "White":
                white.append(agent_id)
            elif agent_data["Race of respondent"] == "Black":
                black.append(agent_id)
            
            # edu: college v.s. high school
            edu_yr = int(agent_data["Highest year of school completed"])
            if edu_yr > 12:
                college.append(agent_id)
            if edu_yr <= 12:
                high_school.append(agent_id)
            
            # inc: high (>$55000) v.s. low (<$30000)
            inc_str = agent_data["Total family income"]
            if inc_str == "Under 1000":
                inc_lower_bound, inc_upper_bound = 0, 999
            elif inc_str == "110000 or over":
                inc_lower_bound, inc_upper_bound = 110000, 650000
            else:
                inc_lower_bound, inc_upper_bound = inc_str.split("to")
                inc_lower_bound, inc_upper_bound = int(inc_lower_bound), int(inc_upper_bound)
            if inc_lower_bound >= 55000:
                high.append(agent_id)
            if inc_upper_bound < 30000:
                low.append(agent_id)
        
        race = {"white":white, "black":black}
        edu = {"college":college, "high school":high_school}
        inc = {"high": high, "low":low}
        return race, edu, inc


    def get_agent_info(self):
        return self.agents_data
    

    def get_agent_race_group_ids(self):
        return self.race
    

    def get_agent_edu_group_ids(self):
        return self.edu
    

    def get_agent_inc_group_ids(self):
        return self.inc


class Agent(object):

    def __init__(self, race, edu, inc, inc_unstand, net_size, id) -> None:
        super().__init__()
        self.reser_price = None
        self.have_bought = False
        self.net_perc = 0 # percentage of adopters
        self.spec_net_list = list()

        self.race = float(race)
        self.edu = float(edu)
        self.inc = float(inc)
        self.inc_unstand = float(inc_unstand)
        self.net_size = int(net_size)
        self.id = int(id)


    def update_reser_price(self, args):
        self.reser_price = (args.k * self.inc_unstand**args.gamma
            + self.inc_unstand**args.gamma*args.delta*self.net_perc**args.alpha)
    

    def update_spec_net_perc(self):
        if not self.spec_net_list:
            self.net_perc = 0
        else:
            list_of_bought = [agent for agent in self.spec_net_list if agent.have_bought]
            self.net_perc = len(list_of_bought) / len(self.spec_net_list)
    

    def update_general_net_perc(self, new_net_perc):
        self.net_perc = new_net_perc
    

    def want_adopt_internet(self, market_price):
        if self.reser_price is None:
            raise ValueError("The reservation price is not initialized.")
        return (market_price <= self.reser_price)
    

    def get_social_status(self):
        return np.array([self.race, self.edu, self.inc])
    

    def get_id(self):
        return self.id
    

    def get_net_size(self):
        return self.net_size
    

    def tie_with(self, agent):
        """ agent -> Agent: should be an pointer to an Agent object. """
        self.spec_net_list.append(agent)


class InternetModel(object):

    class Logger(object):
        def __init__(self, ids_dict, keys) -> None:
            super().__init__()
            self.ids_dict = ids_dict
            self.keys = keys

            self.key1_n = 0
            self.key2_n = 0
            self.key1_perc = list()
            self.key2_perc = list()
            self.key1_key2_odd_ratio = list()
        

        def log_id(self, ag_id):
            if ag_id in self.ids_dict[self.keys[0]]:
                self.key1_n += 1
            if ag_id in self.ids_dict[self.keys[1]]:
                self.key2_n += 1


        def log_into_list(self):
            self.key1_perc.append(self.key1_n/len(self.ids_dict[self.keys[0]]))
            self.key2_perc.append(self.key2_n/len(self.ids_dict[self.keys[1]]))
            self.key1_key2_odd_ratio.append(self.key1_n/self.key2_n if self.key2_n else 0)
        

        def get_latest_logged(self):
            return "{}: {:.2f}%; {}:{:.2f}%; odd_r: {:.2f}".format(self.keys[0], self.key1_perc[-1],
                self.keys[1], self.key2_perc[-1], self.key1_key2_odd_ratio[-1])
                

    def __init__(self, args, data_holder:AgentDataHolder) -> None:
        super().__init__()
        self.args = args
        self.data_holder = data_holder
        self.dis_w = np.array([args.w_race, args.w_edu, args.w_inc])

        self.adopters_n = 0
        self.new_adopters_n = 0
        self.adopters_perc = list()
        self.internet_price = args.p_0
        self.period = 0

        self.agents = None
        self.race_logger = self.Logger(data_holder.get_agent_race_group_ids(),
                                       keys=["white", "black"])
        self.edu_logger = self.Logger(data_holder.get_agent_edu_group_ids(),
                                      keys=["college", "high school"])
        self.inc_logger = self.Logger(data_holder.get_agent_inc_group_ids(),
                                      keys=["high", "low"])
        
        print("Args: {}".format(self.args))
        self.preparation_phrase()
    

    def preparation_phrase(self):
        # 1. Build N Agents
        self.agents, self.agent_n = self.build_agents()

        # 2. Build a network if the identity-specific network is enabled
        if self.args.is_spec_net:
            agent_dis_matrix = self.build_agent_dis_matrix()
            # find ego_net
            print("building ego net for each agent ...")
            for agent_idx, agent in enumerate(self.agents):
                ingroup_n = int(agent.get_net_size()*self.args.scaler)
                sorted_idx = np.argsort(agent_dis_matrix[agent_idx, :])
                ingroup_agent_idx = sorted_idx[:ingroup_n]
                outgroup_agent_idx = sorted_idx[ingroup_n:]
                prob_to_ingroup = self.args.h + (1-self.args.h)*np.random.uniform()
                for _ in range(agent.get_net_size()):
                    prob = np.random.uniform()
                    # tie with out-group
                    if prob > prob_to_ingroup:
                        agent.tie_with(self.agents[np.random.choice(outgroup_agent_idx)])
                    # tie with in-group
                    elif prob <= prob_to_ingroup:
                        agent.tie_with(self.agents[np.random.choice(ingroup_agent_idx)])
        
        # 3. Initialize Agents’ reservation price
        for agent in self.agents:
            agent.update_reser_price(self.args)
        
        print("Model finished initialization and preparation.")


    def build_agents(self):
        agents = list()
        agents_data = self.data_holder.get_agent_info()
        for agent_data in agents_data:
            agent = Agent(id=agent_data[0],
                          net_size=agent_data[1],
                          race=agent_data[2],
                          edu=agent_data[3],
                          inc=agent_data[4],
                          inc_unstand=agent_data[5])
            agents.append(agent)
        agent_n = len(agents)
        print("{} agents initialized.".format(agent_n))
        return agents, agent_n


    def cal_agents_dis(self, agent1:Agent, agent2:Agent) -> float:
        dis_vector = agent1.get_social_status() - agent2.get_social_status()
        dis = np.dot(dis_vector**2, self.dis_w)
        return dis
    

    def build_agent_dis_matrix(self):
        print("building distance matrix ...")
        dis_matrix = np.full((self.agent_n, self.agent_n), np.inf)
        for i in range(self.agent_n-1):
            for j in range(i+1, self.agent_n):
                dis = self.cal_agents_dis(self.agents[i], self.agents[j])
                dis_matrix[i][j] = dis
                dis_matrix[j][i] = dis
        return dis_matrix
    

    def update_internet_price(self):
        n = self.adopters_n / self.agent_n
        self.internet_price = (self.internet_price + 
            self.args.a/12 * n * (self.args.p_min-self.internet_price))
    

    def simulate_a_period(self):
        if self.agents is None:
            raise ValueError("Call model.preparation_phrase() first.")
        self.period += 1

        # 1. Update the Internet price
        self.update_internet_price()
        self.new_adopters_n = 0
        
        # 2. Agents adopt the Internet
        for ag in self.agents:
            if not ag.have_bought and ag.want_adopt_internet(self.internet_price):
                ag.have_bought = True
                self.adopters_n += 1
                self.new_adopters_n += 1
                self.race_logger.log_id(ag.get_id())
                self.edu_logger.log_id(ag.get_id())
                self.inc_logger.log_id(ag.get_id())

        # 3. Update agent’s percent of adopters
        # 4. Update agent’s reservation price
        if self.args.is_spec_net:
            for ag in self.agents:
                ag.update_spec_net_perc()
                ag.update_reser_price(self.args)
        else:
            general_perc = self.adopters_n / self.agent_n
            for ag in self.agents:
                ag.update_general_net_perc(general_perc)
                ag.update_reser_price(self.args)


        self.adopters_perc.append(self.adopters_n / self.agent_n)
        self.race_logger.log_into_list()
        self.edu_logger.log_into_list()
        self.inc_logger.log_into_list()

        print("period {} || adopters: {}/{}; internet_price: {}".format(self.period,
            self.adopters_n, self.agent_n, self.internet_price))
        print("\t{} || {} || {}".format(self.race_logger.get_latest_logged(),
                                        self.edu_logger.get_latest_logged(),
                                        self.inc_logger.get_latest_logged()))
    

    def simulate(self):
        print("==== START SIMULATION ====")
        for _ in range(self.args.n_period):
            self.simulate_a_period()
        print("==== END SIMULATION ====")


if __name__ ==  "__main__":
    parser = ArgsModel()
    args = parser.get_args()
    np.random.seed(args.seed)

    path_to_agentInfo = os.path.join(os.getcwd(), "agent_info_fil.csv")

    agent_data_holder = AgentDataHolder(path_to_agentInfo)
    internet_model = InternetModel(args, agent_data_holder)
    internet_model.simulate()

    