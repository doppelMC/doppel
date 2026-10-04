//! Ground navigation: a wanted position plus a speed modifier. The
//! walker moves straight toward the want with a 1-block step-up or
//! jump at obstacles. A* arrives in wave 2 behind this same seam.

/// One navigation request. `want` holds the target x/z and the speed
/// modifier; the mob's move control consumes it every tick.
pub struct Nav {
    want: Option<Want>,
    /// Ticks since the request started, for give-up.
    age: i32,
}

struct Want {
    x: f64,
    z: f64,
    modifier: f64,
}

impl Nav {
    pub(crate) fn new() -> Nav {
        Nav { want: None, age: 0 }
    }

    /// Walk toward (x, z) at `modifier` times the base speed. Starts a
    /// fresh give-up clock.
    pub(crate) fn move_to(&mut self, x: f64, z: f64, modifier: f64) {
        self.want = Some(Want { x, z, modifier });
        self.age = 0;
    }

    /// Retargets without resetting the give-up clock.
    pub(crate) fn retarget(&mut self, x: f64, z: f64, modifier: f64) {
        if self.want.is_some() {
            self.want = Some(Want { x, z, modifier });
        } else {
            self.move_to(x, z, modifier);
        }
    }

    pub(crate) fn stop(&mut self) {
        self.want = None;
        self.age = 0;
    }

    /// The active request: target x, target z, speed modifier.
    pub(crate) fn wanted(&self) -> Option<(f64, f64, f64)> {
        self.want.as_ref().map(|w| (w.x, w.z, w.modifier))
    }

    /// Whether the walker stands at the target column.
    pub(crate) fn arrived(&self, x: f64, z: f64) -> bool {
        match &self.want {
            None => true,
            Some(w) => {
                let (dx, dz) = (w.x - x, w.z - z);
                dx * dx + dz * dz < 0.25
            }
        }
    }

    /// Whether a request is active.
    pub(crate) fn in_progress(&self) -> bool {
        self.want.is_some()
    }

    /// One tick of the give-up clock; false past the limit.
    pub(crate) fn tick_age(&mut self, limit: i32) -> bool {
        self.age += 1;
        if self.age > limit {
            self.stop();
            false
        } else {
            true
        }
    }
}

impl Default for Nav {
    fn default() -> Self {
        Self::new()
    }
}
