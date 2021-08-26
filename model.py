import argparse
import csv
import datetime
import os
from PIL.Image import NONE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from plot_3Dmap import Plot3DArray



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
        parser.add_argument('--n_trail', type=int, default=1,
            help="the # of the simulation for each condition.")
        parser.add_argument('--seed', type=int, default=22,
            help="random seed.")
        parser.add_argument('--expNo', type=int, default=1,
            help="the number of the experiments to model one of the 7 conditions.")
        return parser
    

    @staticmethod
    def set_exp_param(args, expNo):
        args.expNo = expNo
        if expNo == 1:
            args.delta = 0.
            args.is_spec_net = False
        elif expNo == 2:
            args.is_spec_net = False
        elif expNo == 3:
            args.is_spec_net = True
            args.h = 0.0
        elif expNo == 4:
            args.is_spec_net = True
            args.h = 0.25
        elif expNo == 5:
            args.is_spec_net = True
            args.h = 0.5
        elif expNo == 6:
            args.is_spec_net = True
            args.h = 0.75
        elif expNo == 7:
            args.is_spec_net = True
            args.h = 1.0
        return args


    def get_args(self):
        args = self.parser.parse_args()
        return self.set_exp_param(args, args.expNo)
    
    
    def get_exp_args(self, expNo):
        args = self.parser.parse_args()
        return self.set_exp_param(args, expNo)


def corr(x, y):
    return np.corrcoef(x, y)[0][1]


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
            [np.array([id, network_size, race_nor, education_nor, income_nor, race, education, income]),
             np.array([id, network_size, race_nor, education_nor, income_nor, race, education, income]),
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
                inc_lower_bound, inc_upper_bound = 110000, 385000
            else:
                inc_lower_bound, inc_upper_bound = inc_str.split("to")
                inc_lower_bound, inc_upper_bound = int(inc_lower_bound), int(inc_upper_bound)
            inc = np.random.randint(inc_lower_bound, inc_upper_bound)
            data.append(float(inc))

            # data
            data = np.array([data])
            agents_data = np.concatenate((agents_data, data), axis=0) if agents_data is not None else data
        
        # cal correlation
        # race v.s. log income
        print("race v.s. log income: {}".format(corr(agents_data[:, 2], np.log(agents_data[:, 4]))))
        print("race v.s. edu: {}".format(corr(agents_data[:, 2], agents_data[:, 3])))
        print("edu v.s. income: {}".format(corr(agents_data[:, 3], agents_data[:, 4])))
        inc = agents_data[:, 4]
        print("income >= 360k: {}".format(inc[inc>=360000].shape))

        # standardize race, education, income
        inc_ori = np.copy(agents_data[:, 2:5]).reshape((-1, 3))
        for idx in range(2, 5):
            fea_arr = np.copy(agents_data[:, idx])
            #agents_data[:, idx] = (fea_arr-np.mean(fea_arr))/np.std(fea_arr)
            agents_data[:, idx] = fea_arr / np.max(fea_arr)
        agents_data = np.concatenate((agents_data, inc_ori), axis=1)

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
        return np.copy(self.agents_data)
    

    def get_agent_race_group_ids(self):
        return self.race
    

    def get_agent_edu_group_ids(self):
        return self.edu
    

    def get_agent_inc_group_ids(self):
        return self.inc


class Agent(object):

    def __init__(self, net_size, id, agent_data_norm, agent_data) -> None:
        super().__init__()
        self.reser_price = None
        self.net_effect = None
        self.have_bought = False
        self.net_perc = 0 # percentage of adopters
        self.spec_net_list = list()

        self.net_size = int(net_size)
        self.id = int(id)
        self.agent_data_norm = agent_data_norm
        self.agent_data = agent_data

        self.inc = self.agent_data[2]


    def update_reser_price(self, args):
        self.reser_price = (args.k * (self.inc**args.gamma)
            + (self.inc**args.gamma)*args.delta*(self.net_perc**args.alpha))
        self.net_effect = (self.inc**args.gamma)*args.delta*(self.net_perc**args.alpha)
    

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
        return self.agent_data_norm
    

    def get_id(self):
        return self.id
    

    def get_net_size(self):
        return self.net_size
    

    def tie_with(self, agent):
        """ agent -> Agent: should be an pointer to an Agent object. """
        self.spec_net_list.append(agent)


class InternetModel(object):
    dis_matrix = None

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
        

        def get_odd_ratio(self):
            return self.key1_key2_odd_ratio
                

    def __init__(self, args, data_holder:AgentDataHolder, plotter3d=None, verbose=True) -> None:
        super().__init__()
        self.args = args
        self.data_holder = data_holder
        self.dis_w = np.array([args.w_race, args.w_edu, args.w_inc])
        self.verbose = verbose
        self.plotter = plotter3d

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
        self.logit_coef = None
        
        if self.verbose:
            print("Args: {}".format(self.args))
        self.preparation_phrase()
    

    def preparation_phrase(self):
        # 1. Build N Agents
        self.agents, self.agent_n = self.build_agents()

        # 2. Build a network if the identity-specific network is enabled
        if self.args.is_spec_net:
            agent_dis_matrix = self.build_agent_dis_matrix()
            # find ego_net
            if self.verbose:
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
                        chosen_ag = np.random.choice(outgroup_agent_idx)
                        outgroup_agent_idx = outgroup_agent_idx[outgroup_agent_idx!=chosen_ag]
                        agent.tie_with(self.agents[chosen_ag])
                    # tie with in-group
                    elif prob <= prob_to_ingroup:
                        chosen_ag = np.random.choice(ingroup_agent_idx)
                        ingroup_agent_idx = ingroup_agent_idx[ingroup_agent_idx!=chosen_ag]
                        agent.tie_with(self.agents[np.random.choice(ingroup_agent_idx)])
        
        # 3. Initialize Agents’ reservation price
        for agent in self.agents:
            agent.update_reser_price(self.args)
        
        if self.verbose:
            print("Model finished initialization and preparation.")


    def build_agents(self):
        agents = list()
        self.agents_data = self.data_holder.get_agent_info()
        for agent_data in self.agents_data:
            agent = Agent(id=agent_data[0],
                          net_size=agent_data[1],
                          agent_data_norm=agent_data[2:5],
                          agent_data=agent_data[5:8])
            agents.append(agent)
        agent_n = len(agents)
        if self.verbose:
            print("{} agents initialized.".format(agent_n))
        
        # for logistic regression
        self.agents_data_norm = np.copy(self.agents_data[:, 2:5])
        self.agents_data_X = np.copy(self.agents_data[:, 5:8])
        self.agents_data_X[:, 2] = np.log(self.agents_data_X[:, 2])
        return agents, agent_n


    def cal_agents_dis(self, agent1:Agent, agent2:Agent) -> float:
        dis_vector = agent1.get_social_status() - agent2.get_social_status()
        weighted_dis = np.linalg.norm(np.multiply(dis_vector, self.dis_w))
        return weighted_dis
    

    def build_agent_dis_matrix(self):
        if InternetModel.dis_matrix is None:
            print("building distance matrix ...")
            InternetModel.dis_matrix = np.full((self.agent_n, self.agent_n), np.inf)
            for i in range(self.agent_n-1):
                for j in range(i+1, self.agent_n):
                    dis = self.cal_agents_dis(self.agents[i], self.agents[j])
                    InternetModel.dis_matrix[i][j] = dis
                    InternetModel.dis_matrix[j][i] = dis
            
        return InternetModel.dis_matrix
    

    def update_internet_price(self):
        n = self.adopters_n / self.agent_n
        self.internet_price = (self.internet_price + 
            self.args.a/12 * n * (self.args.p_min-self.internet_price))
    

    def get_agent_net_effect(self):
        return np.array([ag.net_effect for ag in self.agents])
    

    def get_all_tie(self):
        all_ties = None
        for ag in self.agents:
            for ag_tie in ag.spec_net_list:
                tie = np.concatenate((ag.agent_data_norm.reshape((1, 3)), ag_tie.agent_data_norm.reshape((1, 3))), axis=0).reshape((1, 2, 3))
                all_ties = np.concatenate((all_ties, tie), axis=0) if all_ties is not None else tie
        return all_ties


    def simulate_a_period(self):
        if self.agents is None:
            raise ValueError("Call model.preparation_phrase() first.")
        self.period += 1

        if self.plotter is not None:
            agents_data_X, agents_adp_y = self.get_agent_current_data_norm()
            agents_price = self.get_agent_net_effect()
            agents_tie = self.get_all_tie()
            self.plotter.plot_map(agents_data_X, agents_adp_y, agents_price, agents_tie, self.period)

        # 1. Update the Internet price
        self.update_internet_price()
        
        # 2. Agents adopt the Internet
        for ag in self.agents:
            if not ag.have_bought and ag.want_adopt_internet(self.internet_price):
                ag.have_bought = True
                self.adopters_n += 1
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
        coef = self.logistic_reg()
        self.logit_coef = np.concatenate((self.logit_coef, coef), axis=0) if self.logit_coef is not None else coef

        if self.verbose:
            print("period {} || adopters: {}/{}; internet_price: {}".format(self.period,
                self.adopters_n, self.agent_n, self.internet_price))
            print("\t{} || {} || {}".format(self.race_logger.get_latest_logged(),
                                            self.edu_logger.get_latest_logged(),
                                            self.inc_logger.get_latest_logged()))
    

    def simulate(self):
        if self.verbose:
            print("==== START SIMULATION ====")
        for _ in range(self.args.n_period):
            self.simulate_a_period()
        if self.verbose:
            print("==== END SIMULATION ====")
    

    def get_data_for_plotting(self):
        """
        Return
        - data_concat -> 3d np.array, size=(1, n_period, 4)
            last axis: [percentage of adopters, race odd ratio, education odd ratio, income odd ratio]
        """
        adp_perc = np.array(self.adopters_perc).reshape((1, -1, 1))
        race_odd = np.array(self.race_logger.get_odd_ratio()).reshape((1, -1, 1))
        edu_odd = np.array(self.edu_logger.get_odd_ratio()).reshape((1, -1, 1))
        inc_odd = np.array(self.inc_logger.get_odd_ratio()).reshape((1, -1, 1))
        logit_coef = self.logit_coef.reshape((1, self.args.n_period, 3))
        data_concat = np.concatenate((adp_perc, race_odd, edu_odd, inc_odd, logit_coef), axis=2)
        return data_concat
    
    
    def get_agent_current_data(self):
        agents_adp_y = np.array([(1. if ag.have_bought else 0.) for ag in self.agents])
        return self.agents_data_X, agents_adp_y
    
    def get_agent_current_data_norm(self):
        agents_adp_y = np.array([(1. if ag.have_bought else 0.) for ag in self.agents])
        return self.agents_data_norm, agents_adp_y


    def logistic_reg(self):
        agents_data_X, agents_adp_y = self.get_agent_current_data()
        log_model = LogisticRegression(random_state=args.seed,
                                       class_weight=None)
        log_model.fit(agents_data_X, agents_adp_y)
        coef = log_model.coef_
        return coef


def visualize_3d(agent_data_holder, expNo, suffix):
    args_exp = parser.get_exp_args(expNo=expNo)
    filename_prefix = "{}_expNo({})".format(suffix, expNo)
    plotter = Plot3DArray(filename_prefix=filename_prefix)

    internet_model = InternetModel(args_exp, agent_data_holder, plotter, verbose=True)
    internet_model.simulate()
    plotter.save_gif()
    plotter.save_mp4()


def run_all_exp(args, agent_data_holder, suffix,
    output_dir=os.path.join(os.getcwd(), "csvfiles")):
    paths_to_csv = list()
    for exp_idx in range(1, 8):
        data_all_trail = None
        args_exp = parser.get_exp_args(expNo=exp_idx)
        print("ExpNo {} | Args: {}".format(exp_idx, args_exp))
        for trail_idx in range(args.n_trail):
            print("ExpNo {} | Trail {}/{}".format(exp_idx, trail_idx+1, args.n_trail))
            internet_model = InternetModel(args_exp, agent_data_holder, verbose=False)
            internet_model.simulate()
            data_a_trail = internet_model.get_data_for_plotting()
            data_all_trail = np.concatenate((data_all_trail, data_a_trail), axis=0) if data_all_trail is not None else data_a_trail
        data_trail_avg = np.mean(data_all_trail, axis=0)

        filen = "{}_expNo{}_adpPerc_raceOdd_eduOdd_incOdd.csv".format(suffix, exp_idx)
        np.savetxt(os.path.join(output_dir, filen), data_trail_avg, delimiter=',',
            header="adpPerc,raceOdd,eduOdd,incOdd, raceCoef, eduCoef, incCoef")
        paths_to_csv.append(os.path.join(output_dir, filen))
        print("data saved to {}".format(os.path.join(output_dir, filen)))
    return paths_to_csv

def read_result(path_to_results, col_n=["adp_perc", "race_odd", "edu_odd", "inc_odd"]):
    data_list = list()
    for path in path_to_results:
        data_list.append(pd.read_csv(path).values)
    adp_perc = np.asarray([exp_data[:,0] for exp_data in data_list])
    race_odd = np.asarray([exp_data[:,1] for exp_data in data_list])
    edu_odd = np.asarray([exp_data[:,2] for exp_data in data_list])
    inc_odd = np.asarray([exp_data[:,3] for exp_data in data_list])
    race_coef = np.asarray([exp_data[:,4] for exp_data in data_list])
    edu_coef = np.asarray([exp_data[:,5] for exp_data in data_list])
    inc_coef = np.asarray([exp_data[:,6] for exp_data in data_list])

    return {"adp_perc":adp_perc,
            "race_odd":race_odd,
            "edu_odd":edu_odd,
            "inc_odd":inc_odd,
            "race_coef":race_coef,
            "edu_coef":edu_coef,
            "inc_coef":inc_coef}

def plot_lines(data, fn, title, legend_n, suffix, xlabel="Period", ylabel="Odds Ratio", add_no_NE=False, figure_size=(9, 9), linewidth=1,
    output_dir=os.path.join(os.getcwd(), "imgfiles")):
    print("fn {} | data_size: {}".format(fn, data.shape))
    plt.figure(figsize=figure_size, dpi=80)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if add_no_NE:
        plt.plot(np.arange(1, data.shape[-1]+1), data[0, :], linewidth=linewidth)
    else:
        legend_n = legend_n[1:]
    for i in range(1, 7):
        plt.plot(np.arange(1, data.shape[-1]+1), data[i, :], linewidth=linewidth)
    plt.legend(legend_n)
    plt.savefig(os.path.join(output_dir, "{}_{}.png".format(suffix, fn)))
    print("fig save to {}".format(os.path.join(output_dir, "{}_{}".format(suffix, fn))))


def plot_result(data_dict, legend_n, suffix):
    plot_lines(data_dict["adp_perc"], "adp_perc", "Proportion of Adopters", 
        legend_n, suffix, ylabel="Proportion of Adopters", add_no_NE=True)
    plot_lines(data_dict["race_odd"], "race_odd", "Odds Ratios of Race (White-Black)", legend_n, suffix)
    plot_lines(data_dict["edu_odd"], "edu_odd", "Odds Ratios of Education (College-High School)", legend_n, suffix)
    plot_lines(data_dict["inc_odd"], "inc_odd", "Odds Ratios of Income (Highest-Lowest)", legend_n, suffix)
    plot_lines(data_dict["race_coef"], "race_coef", "Estimated Coefficient of Race", 
        legend_n, suffix, ylabel="Logit coefficient")
    plot_lines(data_dict["edu_coef"], "edu_coef", "Estimated Coefficient of Education", 
        legend_n, suffix, ylabel="Logit coefficient")
    plot_lines(data_dict["inc_coef"], "inc_coef", "Estimated Coefficient of (Logged) Income", 
        legend_n, suffix, ylabel="Logit coefficient")



if __name__ ==  "__main__":
    parser = ArgsModel()
    args = parser.get_args()
    np.random.seed(args.seed)

    path_to_agentInfo = os.path.join(os.getcwd(), "agent_info_fil.csv")
    agent_data_holder = AgentDataHolder(path_to_agentInfo)
    visualize_3d(agent_data_holder, expNo=3, suffix=datetime.datetime.now().strftime('%m_%d_%H_%M'))
    exit(-1)

    suffix = "{}_ntrail_{}".format(datetime.datetime.now().strftime('%m_%d_%H_%M'), args.n_trail)
    path_to_results = run_all_exp(args, agent_data_holder, suffix)

    #suffix = "08_26_22_04_ntrail_5"
    #path_to_results = [os.path.join(os.getcwd(), "csvfiles", "{}_expNo{}_adpPerc_raceOdd_eduOdd_incOdd.csv".format(suffix,exp_idx)) for exp_idx in range(1, 8)]
    legend_n = ["No NE", "Gen NE", "Spe NE (h=0)", "Spe NE (h=0.25)", "Spe NE (h=0.5)", "Spe NE (h=0.75)", "Spe NE (h=1.0)"]
    data_dict = read_result(path_to_results)
    plot_result(data_dict, legend_n, suffix)

    