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


class Event4FSM(FSM):
    """
    Detect the relaxed sequence:
      1) Brush -> ... -> eat -> ... -> drink
      2) Brush -> ... -> drink -> ... -> eat
    (We ignore any other activities in between the relevant ones.)

    Output event_label (4) whenever you observe that either pattern completes.
    Then reset to look for new potential patterns.
    """
    def __init__(self):
        self.event_label = 4
        self.state = 's4_0'  # Start state: haven't seen 'brush_teeth' yet

    def update_state(self, input):
        """
        Input: activity label at the current time (e.g., 'brush_teeth', 'eat', 'drink', or anything else).
        Returns: 
          - 0 if no pattern is completed at this step
          - 4 if the pattern just completed at this step
        """
        output = 0  # Default: no pattern detected

        if self.state == 's4_0':
            # Waiting for the first "brush"
            if input == 'brush_teeth':
                self.state = 's4_1'
            else:
                # Stay in s4_0, do nothing
                pass

        elif self.state == 's4_1':
            # We have seen a "brush_teeth"; looking for "eat" or "drink"
            if input == 'eat':
                self.state = 's4_2'
            elif input == 'drink':
                self.state = 's4_3'
            elif input == 'brush_teeth':
                # Still in the "have brushed" mode
                self.state = 's4_1'
            else:
                # Ignore other activities
                pass

        elif self.state == 's4_2':
            # Pattern so far: brush -> ... -> eat
            # We are waiting for "drink" to complete the pattern
            if input == 'drink':
                # Pattern complete: brush -> eat -> drink
                output = self.event_label  # Trigger detection
                self.state = 's4_0'       # Reset
            elif input == 'brush_teeth':
                # Possibly a new sequence starting
                self.state = 's4_1'
            else:
                # Remain in s4_2, waiting for "drink"
                pass

        elif self.state == 's4_3':
            # Pattern so far: brush -> ... -> drink
            # We are waiting for "eat" to complete the pattern
            if input == 'eat':
                # Pattern complete: brush -> drink -> eat
                output = self.event_label
                self.state = 's4_0'
            elif input == 'brush_teeth':
                # Possibly a new sequence starting
                self.state = 's4_1'
            else:
                # Remain in s4_3, waiting for "eat"
                pass

        return output
    

class Event5FSM(FSM):
    """
    Detect the relaxed sequence: sit -> * -> (type or click) -> * -> walk.
    Both * positions allow ANY activities (including 'sit') until the needed next activity is found.
    
    Output event_label (5) whenever the pattern completes.
    Then reset to look for a new pattern.
    """
    def __init__(self):
        self.event_label = 5
        self.state = 's5_0'  # Initial state: haven't seen 'sit' yet

    def update_state(self, input):
        """
        Input: activity label at the current time (e.g., 'sit', 'walk', 'type', 'click_mouse', etc.).
        Returns:
          - 0 if no pattern completes at this step
          - 5 if the pattern just completed at this step
        """
        output = 0  # Default: no event triggered

        if self.state == 's5_0':
            # Looking for the first "sit"
            if input == 'sit':
                self.state = 's5_1'
            # Otherwise remain in s5_0

        elif self.state == 's5_1':
            # We have seen "sit" -> ... 
            # Waiting for "type" or "click_mouse"
            if input in ('type', 'click_mouse'):
                self.state = 's5_2'
            # Ignore anything else (including another sit) and stay in s5_1

        elif self.state == 's5_2':
            # We have: sit -> ... -> (type or click) -> ...
            # Waiting for "walk"
            if input == 'walk':
                # Pattern complete
                output = self.event_label
                # Reset
                self.state = 's5_0'
            else:
                # Ignore everything else (including 'sit', 'type', 'click_mouse', or anything)
                pass

        return output


class Event6FSM(FSM):
    """
    Detect when washing lasts for 30 seconds consecutively.
    We assume each time step is 5 seconds, so 30 seconds = 6 consecutive 'wash' windows.

    Once we detect 6 consecutive 'wash' inputs, output event_label (6)
    immediately and reset the FSM.
    """
    def __init__(self):
        self.event_label = 6
        self.state = 's6_0'
        self.wash_count = 0  # How many consecutive 'wash' windows

    def update_state(self, input):
        """
        Input: activity label at the current time window (e.g., 'wash', 'type', etc.)
        Returns:
          - 0 if no event is triggered at this step
          - 6 if we just detected 30s consecutive washing
        """
        output = 0  # Default: no event

        if self.state == 's6_0':
            # Not counting washes yet
            if input == 'wash':
                # Start counting
                self.wash_count = 1
                self.state = 's6_1'
            else:
                # Remain idle in s6_0
                pass

        elif self.state == 's6_1':
            # We are counting consecutive 'wash' windows
            if input == 'wash':
                self.wash_count += 1
                if self.wash_count >= 6:
                    # 6 consecutive washes = 30 seconds
                    output = self.event_label
                    # Reset
                    self.wash_count = 0
                    self.state = 's6_0'
            else:
                # The streak is broken, reset
                self.wash_count = 0
                self.state = 's6_0'

        return output
    

class Event7FSM(FSM):
    """
    Adequate Brushing Time (label=7):
    
    - Trigger event 7 when total brushing time reaches 120 seconds (24 windows of 5s).
    - Timer pauses if brushing stops.
    - If brushing resumes within 10 seconds, continue from old total.
    - If brushing does not resume within 10 seconds, reset the total.
    - Once 2-minute threshold is reached, event is reported and timer resets.
    """

    def __init__(self):
        self.event_label = 7
        self.state = 's7_0'
        self.total_brushing_time = 0  # in seconds
        self.gap_time = 0             # how long since brushing stopped, in seconds

    def update_state(self, input):
        """
        Input:  activity label for this 5-second window (e.g. 'brush_teeth' or anything else).
        Return: 0 if no event triggered, or 7 if "adequate brushing" threshold reached this step.
        """
        output = 0

        if self.state == 's7_0':
            # Idle state
            if input == 'brush_teeth':
                self.total_brushing_time = 5
                self.state = 's7_1'
            # else remain in s7_0

        elif self.state == 's7_1':
            # Brushing state
            if input == 'brush_teeth':
                self.total_brushing_time += 5
                # Check threshold
                if self.total_brushing_time >= 120:
                    # Adequate brushing
                    output = self.event_label
                    # Reset everything
                    self.total_brushing_time = 0
                    self.gap_time = 0
                    self.state = 's7_0'
            else:
                # Brushing stopped
                self.gap_time = 5
                self.state = 's7_2'

        elif self.state == 's7_2':
            # Paused after brushing stopped
            if input == 'brush_teeth':
                if self.gap_time <= 10:
                    # Resume brushing
                    self.total_brushing_time += 5
                    self.state = 's7_1'
                    # Check threshold immediately
                    if self.total_brushing_time >= 120:
                        output = self.event_label
                        # Reset
                        self.total_brushing_time = 0
                        self.gap_time = 0
                        self.state = 's7_0'
                else:
                    # Gap too long, reset
                    self.total_brushing_time = 0
                    self.gap_time = 0
                    self.state = 's7_0'
            else:
                # Still not brushing
                self.gap_time += 5
                if self.gap_time > 10:
                    # Gap exceeded
                    self.total_brushing_time = 0
                    self.gap_time = 0
                    self.state = 's7_0'
                # else remain in s7_2

        return output


class Event8FSM(FSM):
    """
    Post-Meal Rest (label=8):
    
    - After 'eat', we track how many 5s-intervals have passed.
    - We store this in an integer counter 'time_counter' from 0..36 (36 = 180s).
    - If user starts working (type/click_mouse) and time_counter >= 36, we trigger event 8.
    - On seeing 'eat', we reset the counter to 0 (restart the "post-meal" clock).
    - After we trigger event 8 or see them working too early, we reset the FSM to s8_0.
    
    This is a finite-state approach because we store only a bounded integer (up to 36), 
    rather than an unbounded timer.
    """

    def __init__(self):
        self.event_label = 8
        # Two states:
        #   's8_0': not tracking (haven't eaten recently)
        #   's8_1': have eaten, counting up to 36 intervals of 5s
        self.state = 's8_0'
        self.time_counter = 0  # 0..36, each step = 5s, 36 = 180s

    def update_state(self, input):
        """
        :param input_activity: the activity in the current 5s window (e.g. 'eat', 'type', 'click_mouse', etc.).
        :return: 0 if no event triggered, or 8 if "post-meal rest" event is detected.
        """
        output = 0  # Default

        if self.state == 's8_0':
            # Not tracking yet, waiting for 'eat'
            if input == 'eat':
                self.time_counter = 0
                self.state = 's8_1'
            # else stay in s8_0 doing nothing

        elif self.state == 's8_1':
            # We have eaten, and are counting the intervals since the last meal
            if input == 'eat':
                # Reset the post-meal counter to 0 (new meal started or continuing)
                self.time_counter = 0
                # Stay in s8_1
            elif input in ('type', 'click_mouse'):
                # User starts working
                if self.time_counter >= 36:
                    # They waited >= 3 minutes (36 intervals)
                    output = self.event_label
                # Either way, once they work, we reset
                self.state = 's8_0'
                self.time_counter = 0
            else:
                # Some other activity (walk, wash, flush, etc.) or idle
                # Increment the counter by 1 step of 5s, capped at 36
                if self.time_counter < 36:
                    self.time_counter += 1
                # If already 36, remain 36 (meaning ">= 180s")

        return output

    
class Event9FSM(FSM):
    """
    Active Typing (label=9):
    
    Detect if three "typing sessions" begin within 60s of the first session's start.
    We have only one input label "type" (vs. any other label).

    Each step = 5s, so 12 intervals = 60s.

    - 'idle': not currently typing
    - 'typing_session': consecutive windows of 'type' (the session remains ongoing)

    session_count: how many sessions have started so far in the current window
    time_since_first_session_start: increments (in 5s steps) while session_count is 1 or 2
    """

    def __init__(self):
        self.event_label = 9

        self.state = 'idle'
        self.session_count = 0

        # Number of 5s intervals since first session *started* (0..12)
        self.time_since_first_session_start = 0

    def update_state(self, input):
        """
        :param activity: 'type' if user is typing, or any other string if not typing.
        :return: 0 if no event, or 9 if "3 sessions start within 60s" is detected.
        """
        output = 0

        # 1) Time progression for the "60s window"
        #    Only track time if we've begun the first session but haven't started the third
        if 1 <= self.session_count < 3:
            self.time_since_first_session_start += 1
            if self.time_since_first_session_start > 12:
                # More than 60s passed -> can't achieve 3 sessions in time
                self._reset_fsm()
                return output  # No event

        # 2) State transitions
        if self.state == 'idle':
            if input == 'type':
                # A new session starts right now
                self.state = 'typing_session'
                self.session_count += 1

                # If this is the first session, start the clock at 0
                if self.session_count == 1:
                    self.time_since_first_session_start = 0

                # If this is the third session
                if self.session_count == 3:
                    # Check if still within 60s
                    if self.time_since_first_session_start <= 12:
                        output = self.event_label
                    # Reset either way
                    self._reset_fsm()
                    return output
            else:
                # Remain idle
                pass

        elif self.state == 'typing_session':
            # If the user stops typing, the session ends
            if input != 'type':
                self.state = 'idle'
            else:
                # Still typing => same session
                pass
        
        return output

    def _reset_fsm(self):
        """Reset everything to initial state."""
        self.state = 'idle'
        self.session_count = 0
        self.time_since_first_session_start = 0


class Event10FSM(FSM):
    """
    Focused Work Start (label=10):
    
    - A "working session" starts with the first 'sit' and ends with the first 'walk'.
    - We count 'click_mouse' events during a session.
    - Once we reach 10 clicks, we trigger event=10 (only once) for that session,
      then ignore any further clicks until the user walks.
    - If the user walks at any time, that ends the session and resets everything.
    """

    def __init__(self):
        self.event_label = 10

        # States:
        #   s10_0 : not in a working session
        #   s10_1 : in a working session, haven't triggered event 10
        #   s10_2 : in a working session, already triggered event 10
        self.state = 's10_0'
        self.click_count = 0

    def update_state(self, input):
        """
        :param activity: e.g. 'sit', 'walk', 'click_mouse', 'type', etc.
        :return: 0 if no event triggered, or 10 if "Focused Work Start" is detected.
        """
        output = 0  # Default no event

        if self.state == 's10_0':
            # Not in a session
            if input == 'sit':
                # Start a working session
                self.click_count = 0
                self.state = 's10_1'
            # else remain in s10_0

        elif self.state == 's10_1':
            # In session, event not triggered
            if input == 'walk':
                # Session ends
                self.state = 's10_0'
                self.click_count = 0
            elif input == 'click_mouse':
                self.click_count += 1
                if self.click_count == 5:
                    # Trigger event 10 once
                    output = self.event_label
                    # Remain in this working session, but switch to s10_2
                    self.state = 's10_2'
            else:
                # Any other activity is allowed, remain in s10_1
                pass

        elif self.state == 's10_2':
            # In session, but event was already triggered
            if input == 'walk':
                # Session ends
                self.state = 's10_0'
                self.click_count = 0
            else:
                # Ignore further clicks or anything else
                pass

        return output



if __name__=="__main__":
    fsm = Event10FSM()

    # Example input sequence:

    # activities = [
    #     'wash', 'wash', 'wash', 'wash', 'wash', 'sit', 'wash',
    #     'walk', 'walk', 'walk', 'sit', 'sit', 'sit', 'sit', 'sit', 'sit', 'sit',
    #     'sit', 'sit', 'sit', 'walk', 'walk', 'walk', 'walk', 'walk', 'walk', 'walk',
    #     'walk', 'walk', 'wash', 'walk', 'walk', 'walk', 'walk', 'walk', 'walk', 'walk',
    #     'walk', 'walk', 'walk', 'walk', 'sit', 'sit', 'walk', 'walk', 'walk', 'walk',
    #     'walk', 'sit', 'eat', 'eat', 'walk', 'eat', 'sit', 'sit', 'sit', 'sit',
    #     'eat', 'walk', 'walk', 'sit', 'walk', 'walk', 'drink', 'drink', 'drink', 'eat'
    # ]

    # activities = [
    #     'sit', 'sit', 'wash', 'wash', 'wash', 'flush_toilet', 'eat', 
    #     'sit', 'sit', 'sit', 'sit', 'sit', 'sit', 'sit', 'sit', 'sit', 'wash',
    #     'sit', 'sit', 'sit', 'sit', 'sit', 'sit', 'sit', 'sit', 'sit', 'wash',
    #     'sit', 'sit', 'sit', 'sit', 'drink', 'sit', 'walk', 'sit', 'sit', 'wash',
    #     'sit', 'sit', 'sit', 'sit', 'sit', 'sit', 'sit', 'sit', 'sit', 'wash',
    #     'sit','click_mouse', 'wash', 'type',
    #     'click_mouse', 'walk', 'click_mouse', 'click_mouse', 'eat', 'sit',  'sit', 'sit', 'walk', 'walk',
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

    # activities = ['walk', 'brush_teeth', 'brush_teeth', 'brush_teeth', 'brush_teeth', 
    #               'brush_teeth',  'brush_teeth',  'brush_teeth',  'brush_teeth',  'brush_teeth',  
    #               'brush_teeth',  'brush_teeth',  'brush_teeth',  'brush_teeth',  'brush_teeth',  
    #               'brush_teeth',  'brush_teeth',  'brush_teeth',  'brush_teeth',  'brush_teeth',  
    #               'brush_teeth',  'brush_teeth',  'brush_teeth', 'walk', 'wash', 'brush_teeth',  'wash', 'brush_teeth',
    #               'wash',  'wash',  'wash', 'eat', 'eat', 'drink', 'eat', 'drink', 'eat',
    #               ]

    activities = [
        'sit', 'sit', 'type', 'wash', 'wash', 'flush_toilet', 'eat',  'type',  'type',  'type', 
        'sit', 'sit', 'sit', 'sit', 'type', 'type', 'sit', 'type', 'type', 'wash',
        'sit', 'sit', 'sit', 'type', 'type', 'click_mouse', 'sit', 'click_mouse', 'sit', 'wash',
        'sit', 'click_mouse', 'click_mouse', 'click_mouse', 'type', 'sit', 'click_mouse', 'sit', 'sit', 'wash',
        'click_mouse', 'click_mouse', 'click_mouse', 'sit', 'sit', 'sit', 'click_mouse', 'walk', 'sit', 'click_mouse',
        'sit','click_mouse', 'click_mouse', 'click_mouse',  'click_mouse',  'type',  'type',  'type',  'type',  'type', 
        'click_mouse', 'sit', 'type', 'click_mouse', 'click_mouse', 'sit',  'click_mouse', 'click_mouse', 'walk', 'walk',
    ]

    states = []
    labels = []
    for i in range(len(activities)):
        states.append(fsm.get_current_state())
        l = fsm.update_state(activities[i])
        labels.append(l)
        if l >0:
            print(i, activities[i], l)

    # print(states)
    print(labels, len(labels))
    