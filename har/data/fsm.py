class FSM():
    def __init__(self) -> None:
        pass

    def get_current_state(self):
        return self.state
    
    def update_state(self, input):
        raise NotImplementedError
    

    
class Event1FSM(FSM):
    def __init__(self):
        super().__init__()
        self.event_label = 1  # For workspace sanitary protocol violation event
        self.state = 's1_0'   # The initial FSM state
        self.wash_counter = 0  # Counter for continuous washing time after restroom use

    def update_state(self, input):
        """
        Input: the activity label input at the current time window.
        Returns: the current event label output - self.event_label if you detect the event of interest, 0 otherwise.
        """
        output = 0  # Default output (no violation detected)

        if self.state == 's1_0':
            if input == 'flush_toilet':
                self.state = 's1_1'      # Transition to state after restroom use
                self.wash_counter = 0    # Reset wash counter

        elif self.state == 's1_1':
            if input == 'wash':
                self.wash_counter += 1   # Increment wash counter
                if self.wash_counter >= 4:
                    self.state = 's1_0'  # Reset to initial state after sufficient washing
            elif input in ('click_mouse', 'type'):
                if self.wash_counter < 4:
                    output = self.event_label  # Violation detected
                self.state = 's1_0'  # Reset state
            else:
                self.wash_counter = 0     # Reset wash counter if any other activity occurs
                # Remain in 's1_1' state

        return output



   



class Event2FSM(FSM):
    def __init__(self):
        super().__init__()
        self.event_label = 2  # For hygiene eating habits protocol violation event
        self.state = 's2_0'  # The initial FSM state
        self.wash_count = 0  # Counter for consecutive wash activities
        self.time_since_wash = float('inf')  # Time counter since last wash

    def update_state(self, input):
        """
        Input: the activity label input at the current time window.
        Returns: the current event label output - self.event_label if you detect the event of interest, 0 otherwise.
        """
        output = 0  # Default output (no violation detected)

        # Check if the input is a meal activity (eating or drinking)
        is_meal = input in ['eat', 'drink']

        if self.state == 's2_0':  # Initial state or "hands need washing" state
            self.wash_count = 0
            if input == 'wash':
                self.state = 's2_1'
                self.wash_count = 1
            elif is_meal:
                self.state = 's2_3'
                output = self.event_label  # Violation detected

        elif self.state == 's2_1':  # "Washing hands" state, hands are not clean yet
            if input == 'wash':
                self.wash_count += 1
                if self.wash_count >= 4:  # 20 seconds (4 * 5-second windows)
                    self.state = 's2_2'
                    self.time_since_wash = 0
            elif is_meal:
                self.state = 's2_3'
                self.wash_count = 0
                output = self.event_label  # Violation detected
            else:
                self.state = 's2_0'
                self.wash_count = 0

        elif self.state == 's2_2':  # "Clean hands" state
            if is_meal:
                self.state = 's2_3'  # Eating or drinking is allowed, stay in clean hands state
            elif input in ['brush_teeth', 'click_mouse', 'flush_toilet', 'type']:
                self.state = 's2_0'  # Touching other things, need to wash hands again
            elif input == 'wash':
                self.time_since_wash = 0  # Reset timer, but stay in clean hands state
            else:
                self.time_since_wash += 1

            if self.time_since_wash > 24:  # More than 2 minutes (24 * 5-second windows)
                self.state = 's2_0'

        elif self.state == 's2_3':  # "Having meals" state， stop the timer 
            if is_meal or input == 'sit': # If eat, drink and sit, then stay in meal state
                pass
            elif input in ['brush_teeth', 'click_mouse', 'flush_toilet', 'type']:
                self.state = 's2_0'  # Touching other things, need to wash hands again
            elif input == 'wash':
                self.time_since_wash = 0  # Reset timer and exit "Having meals" stage
                if self.wash_count >= 4:
                    self.state = 's2_2' # Go back to clean hands state
                else: 
                    self.state = 's2_1'
            else: # Exit the meal state
                if self.wash_count >= 4: # Continue the timer and go back to clean hands state
                    self.time_since_wash += 1
                    self.state = 's2_2'
                else:
                    self.state = 's2_0'

        return output



    
class Event3FSM(FSM):
    def __init__(self):
        super().__init__()
        self.event_label = 3  # For brushing teeth for not enough time event
        self.state = 's3_0'   # Initial FSM state
        self.total_brushing_time = 0  # Total time spent brushing teeth
        self.gap_time = 0  # Time since brushing stopped

    def update_state(self, input):
        """
        Input: the activity label input at the current time window.
        Returns: the current event label output - self.event_label if you detect the event of interest, 0 otherwise.
        """
        output = 0  # Default output (no violation detected)

        if self.state == 's3_0':
            if input == 'brush_teeth':
                # Start brushing
                self.total_brushing_time = 5
                self.state = 's3_1'
            else:
                # Remain in s3_0
                pass

        elif self.state == 's3_1':
            if input == 'brush_teeth':
                # Continue brushing
                self.total_brushing_time += 5
                # Remain in s3_1
            else:
                # Brushing has stopped
                self.gap_time = 5
                self.state = 's3_2'

        elif self.state == 's3_2':
            if input == 'brush_teeth':
                # Brushing resumed
                self.total_brushing_time += 5
                self.gap_time = 0
                self.state = 's3_1'
            else:
                self.gap_time += 5
                if self.gap_time > 10:
                    # Brushing has stopped for more than 10 seconds
                    if self.total_brushing_time < 120:
                        output = self.event_label
                    # Reset variables and transition back to s3_0
                    self.total_brushing_time = 0
                    self.gap_time = 0
                    self.state = 's3_0'
                else:
                    # Remain in s3_2
                    pass
        return output



if __name__=="__main__":
    # Example input sequence:
    fsm = Event3FSM()
    # activities = [
    #     'wash', 'wash', 'wash', 'wash',
    #     'walk', 'walk', 'walk', 'sit', 'sit', 'sit', 'sit', 'sit', 'sit', 'sit',
    #     'sit', 'sit', 'sit', 'walk', 'walk', 'walk', 'walk', 'walk', 'walk', 'walk',
    #     'walk', 'walk', 'wash', 'walk', 'walk', 'walk', 'walk', 'walk', 'walk', 'walk',
    #     'walk', 'walk', 'walk', 'walk', 'sit', 'sit', 'walk', 'walk', 'walk', 'walk',
    #     'walk', 'sit', 'eat', 'eat', 'walk', 'eat', 'sit', 'sit', 'sit', 'sit',
    #     'eat', 'walk', 'walk', 'sit', 'walk', 'walk', 'drink', 'drink', 'drink', 'eat'
    # ]
    # activities = [
    #     'sit', 'sit', 'wash', 'wash', 'wash', 'flush_toilet', 'wash', 'type', 'wash', 'type',
    #     'walk', 'walk', 'click_mouse', 'click_mouse', 'type', 'eat',  'eat', 'eat', 'walk', 'walk',
    # ]
    # activities = [
    #     'eat', 'eat', 'eat',  'eat', 'eat', 'sit', 'wash', 'eat'
    # ]
    # activities = ['walk', 'walk', 'walk', 'walk', 'eat', 'eat', 'eat', 'eat', 'sit', 'sit', 
    #               'sit', 'drink', 'drink', 'sit', 'sit', 'eat', 'wash', 'wash', 'wash', 'eat', 
    #               'eat', 'sit', 'sit', 'eat', 'eat', 'sit', 'sit', 'eat', 'eat', 'drink', 
    #               'walk', 'walk', 'walk', 'walk', 'walk', 'walk', 'walk'
    #               ]
    # activities = ['brush_teeth', 'brush_teeth', 'brush_teeth', 'brush_teeth', 'brush_teeth', 'brush_teeth', 'wash', 'wash', 'brush_teeth', 'brush_teeth', 
    #               'brush_teeth', 'brush_teeth', 'brush_teeth', 'brush_teeth', 'brush_teeth', 'brush_teeth', 'brush_teeth', 'brush_teeth', 'brush_teeth', 'brush_teeth', 
    #               'brush_teeth', 'brush_teeth', 'brush_teeth', 'brush_teeth', 'brush_teeth', 'brush_teeth', 'walk', 
    #               'walk', 'walk', 'walk', 'walk', 'walk', 'walk', 'walk', 'brush_teeth', 'brush_teeth',
    #               ]
    activities = ['walk', 'brush_teeth', 'brush_teeth', 'brush_teeth', 'brush_teeth', 
                  'brush_teeth',  'brush_teeth',  'brush_teeth',  'brush_teeth',  'brush_teeth',  
                  'brush_teeth',  'brush_teeth',  'brush_teeth',  'brush_teeth',  'brush_teeth',  
                  'brush_teeth',  'brush_teeth',  'brush_teeth',  'brush_teeth',  'brush_teeth',  
                  'brush_teeth',  'brush_teeth',  'brush_teeth',  'brush_teeth',  'wash',
                  'wash',  'wash',  'wash',  'walk',  'walk',
                  ]
    

    states = []
    labels = []
    for i in activities:
        l = fsm.update_state(i)
        labels.append(l)
        print(i, l)



    # output = sanitary_eating_violation_detector(activities)
    print(states)
    print(labels, len(labels))
    
    # print(output == labels)
