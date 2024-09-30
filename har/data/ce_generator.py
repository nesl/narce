import numpy as np
from datetime import datetime
from stages import *
from activities import *


def convert_time(time):
        t = datetime.strptime(time, "%H:%M").time()
        t_second = (t.hour * 60 + t.minute) * 60 + t.second
        return t_second


class CEGenerator():
    def __init__(self, n_data, start_time, end_time, time_unit=5):
        """
        Args:
        """
        self.start_time = convert_time(start_time)
        self.end_time = convert_time(end_time)
        self.n_data = n_data
        self.time_unit = time_unit # This is the window size of multimodal dataset we are going to use
        
        self.total_time_window = (self.end_time - self.start_time)//self.time_unit
        # self.time_window_elapsed = 0
        self.stage_list = []
        self.time_seires = []
        
    def generate_events(self):
        raise NotImplementedError

    # def _get_stage(self, t):
    #     raise NotImplementedError


    def _define_stages(self):
        """
        Define time period of each stage.
        # """
        # self.stage_list.append(DaytimeStage(["06:00", "18:00"])) # default stage before 18:00
        # DailyCareStage(["06:00", "09:00"]) # happen at most 1 time in the time period
        # MealStage(["11:00", "13:00"]) # happen at most 1 time in the time period
        # MealStage(["18:00", "20:00"]) # happen at most 1 time in the time period
        # EveningStage(["18:00","23:00"]) # (need to be after meal), default stage after 18:00
        # DailyCareStage(["20:00", "23:00"]) # happen at most 1 time in the time period


class CE5min(CEGenerator):
    """
    Generate events of only 5 minutes. 
    The absolute time doesn't matter in this case.
    """
    def __init__(self, n_data, datat_cat, start_time="00:00", end_time="00:05", time_unit=5, simple_label=False):
        """
        Args:
            n_data (int): number of data samples to generate.
            datat_cat (string): 'train' or 'test' dataset category.
        """
        super().__init__(n_data, start_time, end_time, time_unit)
        self.datat_cat = datat_cat
        self.simple_label = simple_label
        
    def generate_event(self, event_id):
        """
        x,y, y[t] = a inidates the event a ends at time t
        Event 0: no violation happening
        Event 1: no washing hands after using restroom before other activities (except for walking), or walking away for more than 1 min (1 min window)
        Event 2: no washing hands before meals (re-initialize states related washing if no eating happens in 10mins) (10 min window)
        Event 3: brushing teeth in less than 2 minutes (if no brushing happens in 10 seconds than stop timing for brushing teeth) (2 min window)
        """
        t = self.start_time
        total_time_window = self.total_time_window
        datat_cat = self.datat_cat
        time_window_elapsed = 0

        data_sequence =[]
        event_label_sequence = []
        state_input_sequence = []
        state_output_sequence = []
        action_sequence = []
        action_label_sequence = []

        if event_id == 0:
            #  Event 1
            event1_fsm = Event1FSM()
            restroom_activity = RestroomActivity(datat_cat, enforce_window_length=total_time_window, fsm_list=[event1_fsm], simple_label=self.simple_label)
            action_sequence, data_sequence, action_label_sequence, time_window = restroom_activity.generate_activity()
            event_label_sequence, state_input_sequence, state_output_sequence = restroom_activity.generate_complex_label()
            time_window_elapsed += time_window
            t += time_window_elapsed * self.time_unit
        
        elif event_id == 1:
            #  Event 2
            event2_fsm = Event2FSM()
            meal_activity = HavingMealActivity(datat_cat, enforce_window_length=total_time_window, fsm_list=[event2_fsm], simple_label=self.simple_label)
            action_sequence, data_sequence, action_label_sequence, time_window = meal_activity.generate_activity()
            event_label_sequence, state_input_sequence, state_output_sequence = meal_activity.generate_complex_label()
            time_window_elapsed += time_window
            t += time_window_elapsed * self.time_unit
        
        elif event_id == 2:
            #  Event 3
            event3_fsm = Event3FSM()
            oral_activity = OralCleaningActivity(datat_cat, enforce_window_length=total_time_window, fsm_list=[event3_fsm], simple_label=self.simple_label)
            action_sequence, data_sequence, action_label_sequence, time_window = oral_activity.generate_activity()
            event_label_sequence, state_input_sequence, state_output_sequence = oral_activity.generate_complex_label()
            time_window_elapsed += time_window
            t += time_window_elapsed * self.time_unit

        else:
            raise Exception("event_id out of range.")
        
        return data_sequence, event_label_sequence, state_input_sequence, state_output_sequence, action_sequence, action_label_sequence, time_window_elapsed, t
        
    def generate_CE_dataset(self, is_positive_sample=False, state_mapping=None):
        ce_data = []
        ce_labels = []
        in_states = []
        out_states = []
        ae_labels = []

        n_event = 3
        n_data_per_event = self.n_data // n_event

        for i in range(n_event):
            ce_data_temp = []
            ce_labels_temp = []
            in_states_temp = []
            out_states_temp = []
            ae_labels_temp = []
            if i == n_event - 1:
                n = self.n_data - i * n_data_per_event
            else:
                n = n_data_per_event

            n_zero_data_per_event = n // 2
            n_zero_count = 0

            while len(ce_data_temp) < n:
                
                data_sequence, event_label_sequence, state_input_sequence, state_output_sequence, _, action_label_sequence, _, _ = self.generate_event(i)
                if is_positive_sample and all(label == 0 for label in event_label_sequence):
                    continue
                elif all(label == 0 for label in event_label_sequence):
                    if n_zero_count >= n_zero_data_per_event:
                        continue
                    else:
                        n_zero_count += 1
                elif (len(ce_data_temp) - n_zero_count) >= n - n_zero_data_per_event:
                    continue

                data_sequence = np.concatenate([x[None, ...] for x in data_sequence], axis=0)
                if self.simple_label is True:
                    event_label_sequence[0] = event_label_sequence[0] - 1 # b/c the label must start from 0
                event_label_sequence = np.array(event_label_sequence)
                ce_data_temp.append(data_sequence)
                ce_labels_temp.append(event_label_sequence)

                if state_mapping is not None:
                    state_input_sequence = np.array([state_mapping[item[0]] for item in state_input_sequence])
                    state_output_sequence = np.array([state_mapping[item[0]] for item in state_output_sequence])
                
                in_states_temp.append(state_input_sequence)
                out_states_temp.append(state_output_sequence)

                action_label_sequence = np.array(action_label_sequence)
                ae_labels_temp.append(action_label_sequence)

            ce_data.extend(ce_data_temp)
            ce_labels.extend(ce_labels_temp)
            in_states.extend(in_states_temp)
            out_states.extend(out_states_temp)
            ae_labels.extend(ae_labels_temp)
        
        ce_data = np.concatenate([x[None, ...] for x in ce_data], axis=0)
        ce_labels = np.concatenate([x[None, ...] for x in ce_labels], axis=0)
        in_states = np.stack(in_states, axis=0)
        out_states = np.stack(out_states, axis=0)
        ae_labels = np.stack(ae_labels, axis=0)
        
        return ce_data, ce_labels, ae_labels, in_states, out_states

class CE15min(CE5min):
    """
    Generate events of only 5 minutes. 
    The absolute time doesn't matter in this case.
    """
    def __init__(self, n_data, datat_cat, time_unit=5, simple_label=False):
        """
        Args:
            n_data (int): number of data samples to generate.
            datat_cat (string): 'train' or 'test' dataset category.
        """
        super().__init__(n_data=n_data, datat_cat=datat_cat, start_time="00:00", end_time="00:15", time_unit=time_unit, simple_label=simple_label)
        self.datat_cat = datat_cat
        self.simple_label = simple_label
        
    def generate_event(self, event_id):
        """
        x,y, y[t] = a inidates the event a ends at time t
        Event 0: no violation happening
        Event 1: no washing hands after using restroom before other activities (except for walking), or walking away for more than 1 min (1 min window)
        Event 2: no washing hands before meals (re-initialize states related washing if no eating happens in 10mins) (10 min window)
        Event 3: brushing teeth in less than 2 minutes (if no brushing happens in 10 seconds than stop timing for brushing teeth) (2 min window)
        """
        t = self.start_time
        total_time_window = self.total_time_window
        datat_cat = self.datat_cat
        time_window_elapsed = 0

        data_sequence =[]
        event_label_sequence = []
        state_input_sequence = []
        state_output_sequence = []
        action_sequence = []
        action_label_sequence = []

        if event_id == 0:
            #  Event 1
            restroom_action_length_dict = {'walk': (3, 36), # 15s - 3min
                                           'sit': (2, 72),  # 10s - 6min
                                           'wash': (1, 12) # 5s - 1min
                                           }
            restroom_activity_length_dict = {'work': (36, 120), # 3min - 10min
                                             'wander': (48, 120), # 4min - 10min
                                             'walk_in': (12, 24), # 1min - 2min
                                            }
            event1_fsm = Event1FSM()
            restroom_activity = RestroomActivity(datat_cat, enforce_window_length=total_time_window, fsm_list=[event1_fsm], simple_label=self.simple_label, action_length_dict=restroom_action_length_dict, activity_length_dict=restroom_activity_length_dict)
            action_sequence, data_sequence, action_label_sequence, time_window = restroom_activity.generate_activity()
            event_label_sequence, state_input_sequence, state_output_sequence = restroom_activity.generate_complex_label()
            time_window_elapsed += time_window
            t += time_window_elapsed * self.time_unit
        
        elif event_id == 1:
            #  Event 2
            meal_action_length_dict = {'walk': (3, 24), # 15s - 2min 
                                       'wash': (1, 12), # 5s - 1min 
                                        }
            meal_activity_length_dict = {'work': (60, 120), # 5min - 10min
                                         'meal': (36, 180), # 3min - 15min
                                        }
            event2_fsm = Event2FSM()
            meal_activity = HavingMealActivity(datat_cat, enforce_window_length=total_time_window, fsm_list=[event2_fsm], simple_label=self.simple_label, action_length_dict=meal_action_length_dict, activity_length_dict=meal_activity_length_dict)
            action_sequence, data_sequence, action_label_sequence, time_window = meal_activity.generate_activity()
            event_label_sequence, state_input_sequence, state_output_sequence = meal_activity.generate_complex_label()
            time_window_elapsed += time_window
            t += time_window_elapsed * self.time_unit
        
        elif event_id == 2:
            #  Event 3
            oral_action_length_dict = {'walk': (3, 72), # 15s - 6min 
                                       'wash': (1, 12), # 5s - 1min 
                                       'brush_teeth': (3, 60), # 15s - 5min
                                       'sit': (12, 60), # 1min - 5min
                                       }
            event3_fsm = Event3FSM()
            oral_activity = OralCleaningActivity(datat_cat, enforce_window_length=total_time_window, fsm_list=[event3_fsm], simple_label=self.simple_label, action_length_dict=oral_action_length_dict)
            action_sequence, data_sequence, action_label_sequence, time_window = oral_activity.generate_activity()
            event_label_sequence, state_input_sequence, state_output_sequence = oral_activity.generate_complex_label()
            time_window_elapsed += time_window
            t += time_window_elapsed * self.time_unit

        else:
            raise Exception("event_id out of range.")
        
        return data_sequence, event_label_sequence, state_input_sequence, state_output_sequence, action_sequence, action_label_sequence, time_window_elapsed, t
        

class CE3min(CE5min):
    """
    Generate events of only 5 minutes. 
    The absolute time doesn't matter in this case.
    """
    def __init__(self, n_data, datat_cat, time_unit=5, simple_label=False):
        """
        Args:
            n_data (int): number of data samples to generate.
            datat_cat (string): 'train' or 'test' dataset category.
        """
        super().__init__(n_data=n_data, datat_cat=datat_cat, start_time="00:00", end_time="00:03", time_unit=time_unit, simple_label=simple_label)
        self.datat_cat = datat_cat
        self.simple_label = simple_label
        
    def generate_event(self, event_id):
        """
        x,y, y[t] = a inidates the event a ends at time t
        Event 0: no violation happening
        Event 1: no washing hands after using restroom before other activities (except for walking), or walking away for more than 1 min (1 min window)
        Event 2: no washing hands before meals (re-initialize states related washing if no eating happens in 2mins) (2 min window)
        Event 3: brushing teeth in less than 2 minutes (if no brushing happens in 10 seconds than stop timing for brushing teeth) (2 min window)
        """
        t = self.start_time
        total_time_window = self.total_time_window
        datat_cat = self.datat_cat
        time_window_elapsed = 0

        data_sequence =[]
        event_label_sequence = []
        state_input_sequence = []
        state_output_sequence = []
        action_sequence = []
        action_label_sequence = []

        if event_id == 0:
            #  Event 1: walk1 -> wash1? -> sitting -> flush -> wash2? -> walk2 ('?' means a random action that may not happen)
            restroom_action_length_dict = {'walk': (1, 2), # 5s - 10s
                                           'sit': (2, 3), # 10s - 15s
                                           }
            restroom_activity_length_dict = {'work': (6, 12), # 30s - 1min
                                             'wander': (6, 24), # 30s - 2min
                                             'walk_in': (1, 3), # 5s - 15s
                                            }
            event1_fsm = Event1FSM()
            restroom_activity = RestroomActivity(datat_cat, enforce_window_length=total_time_window, fsm_list=[event1_fsm], simple_label=self.simple_label, action_length_dict=restroom_action_length_dict, activity_length_dict=restroom_activity_length_dict)
            action_sequence, data_sequence, action_label_sequence, time_window = restroom_activity.generate_activity()
            event_label_sequence, state_input_sequence, state_output_sequence = restroom_activity.generate_complex_label()
            time_window_elapsed += time_window
            t += time_window_elapsed * self.time_unit
        
        elif event_id == 1:
            #  Event 2: wash? -> walk -> (restroom or work)? -> wash? -> meal -> walk2? -> wash? -> meal
            meal_action_length_dict = {'walk': (1, 3), # 5s - 15s 
                                       }
            meal_activity_length_dict = {'work': (6, 12), # 30s - 1min
                                         'meal': (6, 12), # 30s - 1min
                                        }
            event2_fsm = Event2FSM()
            meal_activity = HavingMealActivity(datat_cat, enforce_window_length=total_time_window, fsm_list=[event2_fsm], simple_label=self.simple_label, action_length_dict=meal_action_length_dict, activity_length_dict=meal_activity_length_dict)
            action_sequence, data_sequence, action_label_sequence, time_window = meal_activity.generate_activity()
            event_label_sequence, state_input_sequence, state_output_sequence = meal_activity.generate_complex_label()
            time_window_elapsed += time_window
            t += time_window_elapsed * self.time_unit
        
        elif event_id == 2:
            #  Event 3: sit? -> walk1 -> wash1? -> brush -> wash2 -> (wash + brush)? -> walk2 ('?' means a random action that may not happen)
            oral_action_length_dict = {'walk': (1, 2), # 5s - 10s 
                                       'wash': (1, 3), # 5s - 15s 
                                       'brush_teeth': (3, 36), # 15s - 3min
                                       'sit': (1, 2), # 5s - 10s
                                       }
            event3_fsm = Event3FSM()
            oral_activity = OralCleaningActivity(datat_cat, enforce_window_length=total_time_window, fsm_list=[event3_fsm], simple_label=self.simple_label, action_length_dict=oral_action_length_dict)
            action_sequence, data_sequence, action_label_sequence, time_window = oral_activity.generate_activity()
            event_label_sequence, state_input_sequence, state_output_sequence = oral_activity.generate_complex_label()
            time_window_elapsed += time_window
            t += time_window_elapsed * self.time_unit

        else:
            raise Exception("event_id out of range.")
        
        return data_sequence, event_label_sequence, state_input_sequence, state_output_sequence, action_sequence, action_label_sequence, time_window_elapsed, t
        


class CECombo(CEGenerator):
    """
    Generate multiple events in one data sample. 
    The absolute time doesn't matter in this case.
    """
    def __init__(self, n_data, datat_cat='test', start_time="00:00", end_time="00:15", time_unit=5):
        """
        Args:
            n_data (int): number of data samples to generate.
            datat_cat (string): 'train' or 'test' dataset category.
        """
        super().__init__(n_data, start_time, end_time, time_unit)
        self.datat_cat = datat_cat
        
    def generate_event(self, event_id, total_time_window):
        """
        x,y, y[t] = a inidates the event a ends at time t
        Event 0: no violation happening
        Event 1: no washing hands after using restroom before other activities (except for walking), or walking away for more than 1 min (1 min window)
        Event 2: no washing hands before meals (re-initialize states related washing if no eating happens in 10mins) (10 min window)
        Event 3: brushing teeth in less than 2 minutes (if no brushing happens in 10 seconds than stop timing for brushing teeth) (2 min window)
        """
        t = self.start_time
        datat_cat = self.datat_cat
        time_window_elapsed = 0

        data_sequence =[]
        event_label_sequence = []
        state_input_sequence = []
        state_output_sequence = []
        action_sequence = []
        action_label_sequence = []

        if event_id == 0:
            #  Event 1
            event1_fsm = Event1FSM()
            restroom_activity = RestroomActivity(datat_cat, enforce_window_length=total_time_window, fsm_list=[event1_fsm], simple_label=self.simple_label)
            action_sequence, data_sequence, action_label_sequence, time_window = restroom_activity.generate_activity()
            event_label_sequence, state_input_sequence, state_output_sequence = restroom_activity.generate_complex_label()
            time_window_elapsed += time_window
            t += time_window_elapsed * self.time_unit
        
        elif event_id == 1:
            #  Event 2
            event2_fsm = Event2FSM()
            meal_activity = HavingMealActivity(datat_cat, enforce_window_length=total_time_window, fsm_list=[event2_fsm], simple_label=self.simple_label)
            action_sequence, data_sequence, action_label_sequence, time_window = meal_activity.generate_activity()
            event_label_sequence, state_input_sequence, state_output_sequence = meal_activity.generate_complex_label()
            time_window_elapsed += time_window
            t += time_window_elapsed * self.time_unit
        
        elif event_id == 2:
            #  Event 3
            event3_fsm = Event3FSM()
            oral_activity = OralCleaningActivity(datat_cat, enforce_window_length=total_time_window, fsm_list=[event3_fsm], simple_label=self.simple_label)
            action_sequence, data_sequence, action_label_sequence, time_window = oral_activity.generate_activity()
            event_label_sequence, state_input_sequence, state_output_sequence = oral_activity.generate_complex_label()
            time_window_elapsed += time_window
            t += time_window_elapsed * self.time_unit

        else:
            raise Exception("event_id out of range.")
        
        return data_sequence, event_label_sequence, state_input_sequence, state_output_sequence, action_sequence, action_label_sequence, time_window_elapsed, t
    
    def generate_CE_dataset(self, is_positive_sample=False, state_mapping=None):
        ce_data = []
        ce_labels = []
        in_states = []
        out_states = []
        ae_labels = []

        n_event = 3
        # n_data_per_event = self.n_data // n_event

        n_combo = 3
        time_window = self.total_time_window // n_combo
        event_list = np.array(range(1, n_event + 1))

        for _ in range(self.n_data):
            event_combo = np.random.choice(event_list, size=n_combo, replace=True)
            for i in event_combo:
                ce_data_temp = []
                ce_labels_temp = []
                in_states_temp = []
                out_states_temp = []
                ae_labels_temp = []
                if i == n_event - 1:
                    n = self.n_data - i * n_data_per_event
                else:
                    n = n_data_per_event

                n_zero_data_per_event = n // 2
                n_zero_count = 0

                while len(ce_data_temp) < n:
                    
                    data_sequence, event_label_sequence, state_input_sequence, state_output_sequence, _, action_label_sequence, _, _ = self.generate_event(i, time_window)
                    if is_positive_sample and all(label == 0 for label in event_label_sequence):
                        continue
                    elif all(label == 0 for label in event_label_sequence):
                        if n_zero_count >= n_zero_data_per_event:
                            continue
                        else:
                            n_zero_count += 1

                    data_sequence = np.concatenate([x[None, ...] for x in data_sequence], axis=0)
                    if self.simple_label is True:
                        event_label_sequence[0] = event_label_sequence[0] - 1 # b/c the label must start from 0
                    event_label_sequence = np.array(event_label_sequence)
                    ce_data_temp.append(data_sequence)
                    ce_labels_temp.append(event_label_sequence)

                    if state_mapping is not None:
                        state_input_sequence = np.array([state_mapping[item[0]] for item in state_input_sequence])
                        state_output_sequence = np.array([state_mapping[item[0]] for item in state_output_sequence])
                    
                    in_states_temp.append(state_input_sequence)
                    out_states_temp.append(state_output_sequence)

                    action_label_sequence = np.array(action_label_sequence)
                    ae_labels_temp.append(action_label_sequence)

                ce_data.extend(ce_data_temp)
                ce_labels.extend(ce_labels_temp)
                in_states.extend(in_states_temp)
                out_states.extend(out_states_temp)
                ae_labels.extend(ae_labels_temp)
        
            ce_data = np.concatenate([x[None, ...] for x in ce_data], axis=0)
            ce_labels = np.concatenate([x[None, ...] for x in ce_labels], axis=0)
            in_states = np.stack(in_states, axis=0)
            out_states = np.stack(out_states, axis=0)
            ae_labels = np.stack(ae_labels, axis=0)
        
        return ce_data, ce_labels, ae_labels, in_states, out_states

if __name__ == '__main__':
    n_data = 20
    ce5 = CE5min(n_data, 'train')
    action_data, event_labels, in_states, out_states, actions, action_labels, windows, t = ce5.generate_event(0)
    # for a,l in zip(actions, labels):
    #     print(a,l)

    print(event_labels)
    print(in_states)
    print(out_states)
    print(actions)
    print(action_labels)

    print(len(action_data))
    print(len(event_labels))
    print(len(in_states))
    print(len(out_states))
    print(len(actions))
    print(len(action_labels))

    print(windows)
    print(t)

    ce_data, ce_labels, ae_labels, in_states, out_states = ce5.generate_CE_dataset()
    print(ce_data.shape, ce_labels.shape, ae_labels.shape)
    print(ce_labels[np.random.randint(n_data)])


